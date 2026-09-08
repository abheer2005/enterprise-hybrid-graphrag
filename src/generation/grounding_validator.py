from collections import OrderedDict
import hashlib
import re
from dataclasses import dataclass
from typing import Any

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEFAULT_NLI_MODEL = (
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)


@dataclass
class ClaimGroundingResult:
    """
    Grounding result for one factual claim.
    """

    claim: str
    citations: list[int]

    supported: bool

    entailment_score: float
    contradiction_score: float
    neutral_score: float

    label: str

    evidence_ids: list[int]


@dataclass
class GroundingValidationResult:
    """
    Overall grounding validation result.
    """

    passed: bool

    claims: list[ClaimGroundingResult]

    supported_claims: list[
        ClaimGroundingResult
    ]

    unsupported_claims: list[
        ClaimGroundingResult
    ]


class GroundingValidator:
    """
    Claim-level NLI grounding validator.

    Purpose:
        Verify whether each factual claim made by the
        generated answer is supported by the documentary
        evidence cited by that claim.

    Design goals:
        - Catch fabricated claims.
        - Catch contradictory claims.
        - Accept strong direct support.
        - Accept sufficiently strong paraphrases.
        - Avoid rejecting useful answers merely because
          wording differs from the source.
        - Allow multiple cited chunks to jointly support
          a claim when appropriate.
        - Keep claim-level validation explainable.
        - Perform NLI inference in batches.

    Citation existence/format validation remains the
    responsibility of AnswerValidator.
    """

    CITATION_PATTERN = re.compile(
        r"\[(\d+)\]"
    )

    SENTENCE_SPLIT_PATTERN = re.compile(
        r"(?<=[.!?])\s+|\n+"
    )

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        min_entailment_score: float = 0.70,
        batch_size: int = 4,
        max_length: int = 512,
        adaptive_entailment_score: float = 0.55,
        max_adaptive_contradiction: float = 0.15,
        min_entailment_margin: float = 0.10,
        max_combined_evidence: int = 3,
        local_window_sentences: int = 3,
        max_local_windows_per_evidence: int = 3,
        use_onnx: bool = True,
        onnx_model_dir: str = ".dist/nli_onnx",
        cache_size: int = 4096,
    ):
        self.model_name = model_name

        # Strong direct-support threshold.
        self.min_entailment_score = (
            min_entailment_score
        )

        # Secondary threshold used only when the NLI
        # distribution is otherwise sufficiently safe.
        self.adaptive_entailment_score = (
            adaptive_entailment_score
        )

        # Adaptive support is never allowed when the
        # contradiction probability is substantial.
        self.max_adaptive_contradiction = (
            max_adaptive_contradiction
        )

        # Entailment must clearly beat the next strongest
        # competing class in adaptive mode.
        self.min_entailment_margin = (
            min_entailment_margin
        )

        # Maximum number of cited evidence chunks that can
        # be combined for joint-support evaluation.
        self.max_combined_evidence = max(
            1,
            int(max_combined_evidence),
        )

        # Claim-focused evidence windows reduce false NEUTRAL
        # decisions when direct support is buried in a long chunk.
        self.local_window_sentences = max(1, int(local_window_sentences))
        self.max_local_windows_per_evidence = max(
            1, int(max_local_windows_per_evidence)
        )

        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_size = max(0, int(cache_size))
        self._nli_cache: OrderedDict[str, dict[str, float]] = OrderedDict()

        self.use_onnx = bool(use_onnx)
        self.onnx_model_dir = onnx_model_dir

        print(
            f"Loading NLI grounding model: "
            f"{model_name}"
        )

        if self.use_onnx:

            print(
                "NLI inference backend: ONNX Runtime"
            )

            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    model_name
                )
            )
            self.model = (
                ORTModelForSequenceClassification
                .from_pretrained(
                    self.onnx_model_dir,
                    provider="CPUExecutionProvider",
                )
            )

            # The exported ONNX model is intentionally
            # executed on CPU. This replaces only the
            # inference runtime; validation semantics stay
            # unchanged.
            self.device = torch.device(
                "cpu"
            )

        else:

            print(
                "NLI inference backend: PyTorch"
            )

            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    model_name
                )
            )

            self.model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    model_name
                )
            )

            self.model.eval()

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            self.model.to(
                self.device
            )

        print(
            f"NLI model device: "
            f"{self.device}"
        )

        print(
            f"NLI batch size: "
            f"{self.batch_size}"
        )

        self.label_mapping = (
            self._build_label_mapping()
        )

        print(
            "NLI labels: "
            f"{self.label_mapping}"
        )

        print(
            "NLI grounding thresholds: "
            f"strong={self.min_entailment_score:.2f}, "
            f"adaptive={self.adaptive_entailment_score:.2f}, "
            f"max_contradiction="
            f"{self.max_adaptive_contradiction:.2f}, "
            f"margin={self.min_entailment_margin:.2f}"
        )

    def _build_label_mapping(
        self,
    ) -> dict[str, int]:
        """
        Resolve actual NLI label IDs from the model
        configuration.
        """

        mapping = {}

        id2label = (
            self.model.config.id2label
            or {}
        )

        for label_id, label_name in (
            id2label.items()
        ):

            normalized = str(
                label_name
            ).strip().lower()

            if "entail" in normalized:

                mapping[
                    "entailment"
                ] = int(label_id)

            elif "contrad" in normalized:

                mapping[
                    "contradiction"
                ] = int(label_id)

            elif "neutral" in normalized:

                mapping[
                    "neutral"
                ] = int(label_id)

        required = {
            "entailment",
            "contradiction",
            "neutral",
        }

        missing = (
            required
            - set(mapping.keys())
        )

        if missing:

            raise ValueError(
                "Could not determine NLI label "
                "mapping from model config. "
                f"id2label={id2label}, "
                f"missing={sorted(missing)}"
            )

        return mapping

    @classmethod
    def _extract_citations(
        cls,
        text: str,
    ) -> list[int]:
        """
        Extract citation numbers while preserving
        first-occurrence order.

        Example:

            Claim [1][3]

        returns:

            [1, 3]
        """

        matches = (
            cls.CITATION_PATTERN.findall(
                text or ""
            )
        )

        citation_ids = []
        seen = set()

        for match in matches:

            citation_id = int(
                match
            )

            if citation_id in seen:
                continue

            seen.add(
                citation_id
            )

            citation_ids.append(
                citation_id
            )

        return citation_ids

    @classmethod
    def _remove_citations(
        cls,
        text: str,
    ) -> str:

        text = (
            cls.CITATION_PATTERN.sub(
                "",
                text,
            )
        )

        return " ".join(
            text.split()
        ).strip()

    @classmethod
    def extract_claims(
        cls,
        answer: str,
    ) -> list[dict[str, Any]]:
        """
        Extract citation-bearing claims.

        Missing citations remain the responsibility
        of AnswerValidator.
        """

        if not (
            answer
            and answer.strip()
        ):
            return []

        segments = (
            cls.SENTENCE_SPLIT_PATTERN.split(
                answer
            )
        )

        claims = []

        for segment in segments:

            segment = (
                segment.strip()
            )

            if not segment:
                continue

            citations = (
                cls._extract_citations(
                    segment
                )
            )

            if not citations:
                continue

            claim = (
                cls._remove_citations(
                    segment
                )
            )

            # Remove Markdown bullet prefixes.
            claim = re.sub(
                r"^[\-\*\u2022]+\s*",
                "",
                claim,
            )

            # Remove Markdown heading prefixes.
            claim = re.sub(
                r"^#+\s*",
                "",
                claim,
            )

            # Remove numbered-list prefixes:
            #
            # 1. claim
            # 2) claim
            #
            claim = re.sub(
                r"^\d+[\.\)]\s*",
                "",
                claim,
            )

            claim = (
                claim.strip()
            )

            if not claim:
                continue

            claims.append(
                {
                    "claim": claim,
                    "citations": citations,
                }
            )

        return claims

    @staticmethod
    def _build_evidence_map(
        evidence: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        """
        Build:

            evidence_id -> evidence object
        """

        evidence_map = {}

        for fallback_id, item in enumerate(
            evidence,
            start=1,
        ):

            evidence_id = item.get(
                "evidence_id"
            )

            if evidence_id is None:

                evidence_id = (
                    fallback_id
                )

            try:

                evidence_id = int(
                    evidence_id
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            evidence_map[
                evidence_id
            ] = item

        return evidence_map

    @staticmethod
    def _normalize_evidence_text(
        text: str,
    ) -> str:
        """
        Normalize documentary text before NLI.

        This intentionally performs only conservative
        whitespace cleanup. It does not rewrite source
        meaning.
        """

        if not text:
            return ""

        return " ".join(
            str(text).split()
        ).strip()

    @staticmethod
    def _lexical_terms(text: str) -> set[str]:
        """Extract terms used only to select relevant source windows."""
        return {
            token
            for token in re.findall(r"[A-Za-z0-9]+", (text or "").lower())
            if len(token) >= 3
        }

    @classmethod
    def _split_evidence_sentences(cls, text: str) -> list[str]:
        """Split evidence into sentence-like units without rewriting it."""
        normalized = cls._normalize_evidence_text(text)
        if not normalized:
            return []

        parts = re.split(
            r"(?<=[.!?;:])\\s+|(?=\\(\\s*[a-zA-Z0-9ivxIVX]+\\s*\\))",
            normalized,
        )
        sentences = [part.strip() for part in parts if part and part.strip()]
        return sentences or [normalized]

    def _build_local_evidence_windows(
        self,
        evidence_text: str,
        claim: str,
    ) -> list[str]:
        """
        Select a few claim-relevant windows from cited evidence.

        NLI still decides support. This heuristic only avoids burying
        a directly supporting sentence inside a much longer premise.
        """
        sentences = self._split_evidence_sentences(evidence_text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        claim_terms = self._lexical_terms(claim)
        scored = []

        for index, sentence in enumerate(sentences):
            sentence_terms = self._lexical_terms(sentence)
            overlap = len(claim_terms & sentence_terms)
            coverage = overlap / max(1, len(claim_terms))
            scored.append((coverage, overlap, index))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        selected = [
            index
            for _, overlap, index in scored
            if overlap > 0
        ][: self.max_local_windows_per_evidence]

        if not selected:
            selected = [scored[0][2]]

        radius = self.local_window_sentences // 2
        windows = []
        seen = set()

        for index in selected:
            start = max(0, index - radius)
            end = min(len(sentences), start + self.local_window_sentences)
            start = max(0, end - self.local_window_sentences)
            window = " ".join(sentences[start:end]).strip()

            if window and window not in seen and window != evidence_text:
                seen.add(window)
                windows.append(window)

        return windows

    def _build_combined_evidence(
        self,
        citation_ids: list[int],
        evidence_map: dict[
            int,
            dict[str, Any],
        ],
    ) -> tuple[str, list[int]]:
        """
        Combine a small number of cited evidence chunks.

        Why:
            A generated sentence may summarize information
            spread across multiple cited chunks.

        Important:
            Only evidence explicitly cited by the claim is
            eligible. Uncited retrieval results are never
            silently added.
        """

        texts = []
        used_ids = []

        for citation_id in citation_ids[
            :self.max_combined_evidence
        ]:

            evidence_item = (
                evidence_map.get(
                    citation_id
                )
            )

            if evidence_item is None:
                continue

            evidence_text = (
                self._normalize_evidence_text(
                    evidence_item.get(
                        "text"
                    )
                    or ""
                )
            )

            if not evidence_text:
                continue

            texts.append(
                evidence_text
            )

            used_ids.append(
                citation_id
            )

        if len(texts) < 2:
            return "", []

        combined = "\n\n".join(
            texts
        )

        return combined, used_ids

    def _run_nli_batch(
        self,
        pairs: list[
            tuple[str, str]
        ],
    ) -> list[dict[str, float]]:
        """
        Run NLI over many:

            (premise, hypothesis)

        pairs using batched model inference.

        premise:
            documentary evidence

        hypothesis:
            generated factual claim
        """

        if not pairs:
            return []

        results = []

        for start in range(
            0,
            len(pairs),
            self.batch_size,
        ):

            batch = pairs[
                start:
                start + self.batch_size
            ]

            premises = [
                pair[0]
                for pair in batch
            ]

            hypotheses = [
                pair[1]
                for pair in batch
            ]

            encoded = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            # Temporary profiling only.
            attention_mask = encoded.get(
                "attention_mask"
            )

            if attention_mask is not None:

                lengths = (
                    attention_mask
                    .sum(dim=1)
                    .tolist()
                )

                print(
                    "[NLI PROFILE] "
                    f"Batch: "
                    f"{start // self.batch_size + 1} | "
                    f"Pairs: {len(batch)} | "
                    f"Token lengths: {lengths} | "
                    f"Max: {max(lengths)} | "
                    f"Avg: "
                    f"{sum(lengths) / len(lengths):.1f}",
                    flush=True,
                )

            if not self.use_onnx:

                encoded = {
                    key: value.to(
                        self.device
                    )
                    for key, value
                    in encoded.items()
                }

                with torch.inference_mode():

                    output = self.model(
                        **encoded
                    )

            else:

                output = self.model(
                    **encoded
                )

            logits = output.logits

            if not torch.is_tensor(
                logits
            ):

                logits = torch.as_tensor(
                    logits
                )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=-1,
                )
                .detach()
                .cpu()
            )

            for probability in probabilities:

                entailment = float(
                    probability[
                        self.label_mapping[
                            "entailment"
                        ]
                    ]
                )

                contradiction = float(
                    probability[
                        self.label_mapping[
                            "contradiction"
                        ]
                    ]
                )

                neutral = float(
                    probability[
                        self.label_mapping[
                            "neutral"
                        ]
                    ]
                )

                results.append(
                    {
                        "entailment": entailment,
                        "contradiction": contradiction,
                        "neutral": neutral,
                    }
                )

        return results

    def _is_supported(
        self,
        entailment: float,
        contradiction: float,
        neutral: float,
    ) -> bool:
        """
        Decide whether one NLI evaluation provides
        sufficient documentary support.

        Two support paths are allowed:

        1. Strong support:
           Entailment reaches the normal strict threshold
           and is the strongest class.

        2. Adaptive support:
           Entailment is moderately strong, contradiction
           is low, entailment is the strongest class, and
           it beats the next strongest class by a useful
           margin.

        This is intentionally NOT equivalent to simply
        lowering the global entailment threshold.
        """

        strongest_competitor = max(
            contradiction,
            neutral,
        )

        entailment_is_strongest = (
            entailment > strongest_competitor
        )

        strong_support = (
            entailment
            >= self.min_entailment_score
            and
            entailment_is_strongest
        )

        entailment_margin = (
            entailment
            - strongest_competitor
        )

        adaptive_support = (
            entailment
            >= self.adaptive_entailment_score
            and
            contradiction
            <= self.max_adaptive_contradiction
            and
            entailment_is_strongest
            and
            entailment_margin
            >= self.min_entailment_margin
        )

        return (
            strong_support
            or adaptive_support
        )

    def _classify_evaluation(
        self,
        evaluations: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        Convert one claim's evidence evaluations into
        the final grounding decision.

        Individual and combined cited-evidence
        evaluations are treated uniformly.

        The evaluation with the strongest entailment is
        retained for reporting, while support is decided
        using confidence-aware rules.
        """

        if not evaluations:

            return {
                "supported": False,
                "label": "NO_EVIDENCE",
                "entailment_score": 0.0,
                "contradiction_score": 0.0,
                "neutral_score": 0.0,
                "evidence_ids": [],
            }

        best = max(
            evaluations,
            key=lambda item: (
                item["entailment"]
            ),
        )

        entailment = float(
            best["entailment"]
        )

        contradiction = float(
            best["contradiction"]
        )

        neutral = float(
            best["neutral"]
        )

        supported = (
            self._is_supported(
                entailment=entailment,
                contradiction=contradiction,
                neutral=neutral,
            )
        )

        if supported:

            label = "ENTAILMENT"

        elif (
            contradiction > neutral
            and
            contradiction > entailment
        ):

            label = "CONTRADICTION"

        else:

            label = "NEUTRAL"

        evidence_ids = (
            best.get(
                "evidence_ids"
            )
        )

        if evidence_ids is None:

            evidence_id = (
                best.get(
                    "evidence_id"
                )
            )

            if evidence_id is None:
                evidence_ids = []

            else:
                evidence_ids = [
                    evidence_id
                ]

        # Remove duplicates while preserving order.
        normalized_ids = []
        seen = set()

        for evidence_id in evidence_ids:

            if evidence_id in seen:
                continue

            seen.add(
                evidence_id
            )

            normalized_ids.append(
                evidence_id
            )

        return {
            "supported": supported,
            "label": label,
            "entailment_score": (
                entailment
            ),
            "contradiction_score": (
                contradiction
            ),
            "neutral_score": neutral,
            "evidence_ids": (
                normalized_ids
            ),
        }

    def validate(
        self,
        answer: str,
        evidence: list[dict[str, Any]],
    ) -> GroundingValidationResult:
        """
        Validate all citation-bearing claims.

        Processing:

            1. Extract citation-bearing claims.
            2. Build individual claim/evidence pairs.
            3. Build combined cited-evidence pairs when a
               claim cites multiple chunks.
            4. Run all NLI pairs in batches.
            5. Map NLI results back to claims.
            6. Apply confidence-aware grounding rules.

        Overall grounding passes only when:

            - at least one claim was checked
            - every checked claim is supported
        """

        claims = self.extract_claims(
            answer
        )

        evidence_map = (
            self._build_evidence_map(
                evidence
            )
        )

        # -------------------------------------------------
        # 1. BUILD ALL NLI PAIRS
        # -------------------------------------------------

        nli_pairs = []

        pair_metadata = []

        claim_evaluations: dict[
            int,
            list[dict[str, Any]],
        ] = {
            index: []
            for index in range(
                len(claims)
            )
        }

        for claim_index, claim_data in enumerate(
            claims
        ):

            claim = (
                claim_data["claim"]
            )

            citations = (
                claim_data[
                    "citations"
                ]
            )

            # ---------------------------------------------
            # INDIVIDUAL CITED EVIDENCE
            # ---------------------------------------------

            for citation_id in citations:

                evidence_item = (
                    evidence_map.get(
                        citation_id
                    )
                )

                if evidence_item is None:
                    continue

                evidence_text = (
                    self._normalize_evidence_text(
                        evidence_item.get(
                            "text"
                        )
                        or ""
                    )
                )

                if not evidence_text:
                    continue

                # Evaluate small claim-focused windows first.
                # The original full cited chunk remains as fallback.
                local_windows = self._build_local_evidence_windows(
                    evidence_text=evidence_text,
                    claim=claim,
                )

                for local_window in local_windows:
                    nli_pairs.append(
                        (
                            local_window,
                            claim,
                        )
                    )

                    pair_metadata.append(
                        {
                            "claim_index": claim_index,
                            "evidence_ids": [citation_id],
                            "mode": "local_window",
                        }
                    )

                nli_pairs.append(
                    (
                        evidence_text,
                        claim,
                    )
                )

                pair_metadata.append(
                    {
                        "claim_index": (
                            claim_index
                        ),
                        "evidence_ids": [
                            citation_id
                        ],
                        "mode": "individual",
                    }
                )

            # ---------------------------------------------
            # COMBINED CITED EVIDENCE
            # ---------------------------------------------
            #
            # Example:
            #
            # claim [2][3]
            #
            # If neither chunk alone expresses the whole
            # claim but the two chunks together do, NLI
            # gets one additional joint-evidence test.
            # ---------------------------------------------

            combined_text, combined_ids = (
                self._build_combined_evidence(
                    citation_ids=citations,
                    evidence_map=evidence_map,
                )
            )

            if (
                combined_text
                and combined_ids
            ):

                nli_pairs.append(
                    (
                        combined_text,
                        claim,
                    )
                )

                pair_metadata.append(
                    {
                        "claim_index": (
                            claim_index
                        ),
                        "evidence_ids": (
                            combined_ids
                        ),
                        "mode": "combined",
                    }
                )

        # -------------------------------------------------
        # 2. RUN ALL PAIRS THROUGH NLI IN BATCHES
        # -------------------------------------------------


        print(
            "\n[NLI PROFILE] "
            f"Claims: {len(claims)} | "
            f"Pairs: {len(nli_pairs)} | "
            f"Batch size: {self.batch_size} | "
            f"Batches: "
            f"{(len(nli_pairs) + self.batch_size - 1) // self.batch_size}",
            flush=True,
        )

        # Exact pair caching avoids repeating the dominant CPU cost across
        # retries, repairs, and repeated enterprise questions.
        nli_scores: list[dict[str, float] | None] = [None] * len(nli_pairs)
        missing_pairs = []
        missing_indexes = []
        for index, (premise, hypothesis) in enumerate(nli_pairs):
            key = hashlib.sha256(
                (premise + "\0" + hypothesis).encode("utf-8")
            ).hexdigest()
            cached = self._nli_cache.get(key)
            if cached is not None:
                self._nli_cache.move_to_end(key)
                nli_scores[index] = dict(cached)
            else:
                missing_pairs.append((premise, hypothesis))
                missing_indexes.append((index, key))

        if missing_pairs:
            fresh_scores = self._run_nli_batch(missing_pairs)
            for (index, key), score in zip(missing_indexes, fresh_scores):
                nli_scores[index] = score
                if self.cache_size:
                    self._nli_cache[key] = dict(score)
                    self._nli_cache.move_to_end(key)
                    while len(self._nli_cache) > self.cache_size:
                        self._nli_cache.popitem(last=False)

        nli_scores = [score for score in nli_scores if score is not None]

        if (
            len(nli_scores)
            != len(pair_metadata)
        ):

            raise RuntimeError(
                "NLI result count does not match "
                "claim/evidence pair count."
            )

        # -------------------------------------------------
        # 3. MAP RESULTS BACK TO CLAIMS
        # -------------------------------------------------

        for metadata, scores in zip(
            pair_metadata,
            nli_scores,
        ):
            print("\n" + "=" * 80)
            print("NLI RESULT")
            print(f"Claim Index : {metadata['claim_index']}")
            print(f"Evidence IDs: {metadata['evidence_ids']}")
            print(f"Mode        : {metadata['mode']}")
            print(f"Entailment  : {scores['entailment']:.4f}")
            print(f"Neutral     : {scores['neutral']:.4f}")
            print(f"Contradiction: {scores['contradiction']:.4f}")
            print("=" * 80)

            claim_index = (
                metadata[
                    "claim_index"
                ]
            )

            claim_evaluations[
                claim_index
            ].append(
                {
                    "evidence_ids": (
                        metadata[
                            "evidence_ids"
                        ]
                    ),
                    "mode": (
                        metadata[
                            "mode"
                        ]
                    ),
                    **scores,
                }
            )

        # -------------------------------------------------
        # 4. BUILD CLAIM RESULTS
        # -------------------------------------------------

        results = []

        for claim_index, claim_data in enumerate(
            claims
        ):

            claim = (
                claim_data[
                    "claim"
                ]
            )

            citations = (
                claim_data[
                    "citations"
                ]
            )

            evaluation = (
                self._classify_evaluation(
                    claim_evaluations[
                        claim_index
                    ]
                )
            )

            print("\nFINAL CLAIM DECISION")
            print(f"Claim      : {claim}")
            print(f"Supported  : {evaluation['supported']}")
            print(f"Label      : {evaluation['label']}")
            print(f"Entailment : {evaluation['entailment_score']:.4f}")
            print(f"Neutral    : {evaluation['neutral_score']:.4f}")
            print(f"Contradict : {evaluation['contradiction_score']:.4f}")
            print(f"Evidence   : {evaluation['evidence_ids']}")

            result = (
                ClaimGroundingResult(
                    claim=claim,
                    citations=citations,
                    supported=(
                        evaluation[
                            "supported"
                        ]
                    ),
                    entailment_score=(
                        evaluation[
                            "entailment_score"
                        ]
                    ),
                    contradiction_score=(
                        evaluation[
                            "contradiction_score"
                        ]
                    ),
                    neutral_score=(
                        evaluation[
                            "neutral_score"
                        ]
                    ),
                    label=(
                        evaluation[
                            "label"
                        ]
                    ),
                    evidence_ids=(
                        evaluation[
                            "evidence_ids"
                        ]
                    ),
                )
            )

            results.append(
                result
            )

        # -------------------------------------------------
        # 5. OVERALL DECISION
        # -------------------------------------------------

        supported_claims = [
            result
            for result in results
            if result.supported
        ]

        unsupported_claims = [
            result
            for result in results
            if not result.supported
        ]

        passed = (
            len(results) > 0
            and
            len(unsupported_claims) == 0
        )

        return GroundingValidationResult(
            passed=passed,
            claims=results,
            supported_claims=(
                supported_claims
            ),
            unsupported_claims=(
                unsupported_claims
            ),
        )
