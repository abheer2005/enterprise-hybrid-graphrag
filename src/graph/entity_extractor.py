import os
import re
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


load_dotenv()


EntityType = Literal[
    "ORGANIZATION",
    "POLICY",
    "ROLE",
    "AUTHORITY",
    "LAW_REGULATION",
    "PROCESS",
    "ACTION",
    "REQUIREMENT",
    "DOCUMENT",
    "LOCATION",
    "DATE",
    "CONCEPT",
    "OTHER",
]


class Entity(BaseModel):
    name: str = Field(
        description="Canonical name of the entity."
    )

    type: EntityType

    description: str = Field(
        default="",
        description=(
            "Brief description supported only by the supplied chunk."
        ),
    )


class Relationship(BaseModel):
    source: str

    relation: str = Field(
        description=(
            "Concise semantic predicate such as APPLIES_TO, "
            "REQUIRES, MONITORS, PROTECTED_FROM."
        )
    )

    target: str

    evidence: str = Field(
        description=(
            "Short evidence statement supported by the supplied chunk."
        )
    )


class KnowledgeExtraction(BaseModel):
    entities: list[Entity] = Field(
        default_factory=list
    )

    relationships: list[Relationship] = Field(
        default_factory=list
    )


class KnowledgeExtractor:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from .env"
            )

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-flash-latest",
        )

        self.client = genai.Client(
            api_key=api_key
        )

    @staticmethod
    def normalize_name(name: str) -> str:

        name = re.sub(
            r"\s+",
            " ",
            name,
        ).strip()

        name = name.strip(
            " \t\n\r.,;:"
        )

        # Conservative aliases only.
        # We will build stronger entity resolution separately.
        aliases = {
            "iocl": "Indian Oil Corporation Limited",
            "indianoil": "Indian Oil Corporation Limited",
            "indian oil": "Indian Oil Corporation Limited",
            "indian oil corporation": (
                "Indian Oil Corporation Limited"
            ),
        }

        return aliases.get(
            name.lower(),
            name,
        )

    @staticmethod
    def normalize_relation(
        relation: str
    ) -> str:

        relation = relation.upper().strip()

        relation = re.sub(
            r"[^A-Z0-9]+",
            "_",
            relation,
        )

        relation = re.sub(
            r"_+",
            "_",
            relation,
        )

        return relation.strip("_")



    @staticmethod

    def canonicalize_relationship(
        source: str,
        relation: str,
        target: str,
    ) -> tuple[str, str, str]:

        relation = KnowledgeExtractor.normalize_relation(
            relation
        )

        inverse_relations = {
            "APPROVES": "APPROVED_BY",
            "MONITORS": "MONITORED_BY",
            "GOVERNS": "GOVERNED_BY",
            "REGULATES": "REGULATED_BY",
            "DEFINES": "DEFINED_BY",
            "FORMULATES": "FORMULATED_BY",
        }

        if relation in inverse_relations:
            return (
                target,
                inverse_relations[relation],
                source,
            )

        return (
            source,
            relation,
            target,
        )

    

    def validate_extraction(
        self,
        extraction: KnowledgeExtraction,
    ) -> KnowledgeExtraction:
        """
        Basic structural validation before extracted knowledge
        is allowed to progress toward Neo4j.
        """

        # Remove empty entities.
        valid_entities = []

        seen = set()

        for entity in extraction.entities:

            entity.name = self.normalize_name(
                entity.name
            )

            if len(entity.name) < 2:
                continue

            key = (
                entity.name.lower(),
                entity.type,
            )

            if key in seen:
                continue

            seen.add(key)
            valid_entities.append(entity)

        extraction.entities = valid_entities

        entity_names = {
            entity.name.lower()
            for entity in valid_entities
        }

        valid_relationships = []

        relationship_seen = set()

        for relationship in extraction.relationships:
            
            source = self.normalize_name(
                relationship.source
            )

            target = self.normalize_name(
                relationship.target
            )

            relation = self.normalize_relation(
                relationship.relation
            )

            source, relation, target = (
                self.canonicalize_relationship(
                    source,
                    relation,
                    target,
                )
            )

            relationship.source = source
            relationship.relation = relation
            relationship.target = target

            # Relationship must have all three components.
            if not (
                relationship.source
                and relationship.target
                and relationship.relation
            ):
                continue

            # No self-loop facts for now.
            if (
                relationship.source.lower()
                == relationship.target.lower()
            ):
                continue

            # Both endpoints must actually exist
            # in this extraction.
            if (
                relationship.source.lower()
                not in entity_names
                or relationship.target.lower()
                not in entity_names
            ):
                continue

            key = (
                relationship.source.lower(),
                relationship.relation,
                relationship.target.lower(),
            )

            if key in relationship_seen:
                continue

            relationship_seen.add(key)

            valid_relationships.append(
                relationship
            )

        extraction.relationships = (
            valid_relationships
        )

        return extraction

    def extract(
        self,
        text: str,
    ) -> KnowledgeExtraction:

        system_instruction = """
You are an enterprise knowledge-graph extraction engine.

Your task is NOT to answer questions and NOT to summarize the
document.

Extract only facts explicitly supported by the supplied document
chunk.

STRICT GROUNDING RULES

- Never add knowledge from memory or general world knowledge.
- Never infer a fact merely because it seems likely.
- Never invent missing people, organizations, responsibilities,
  policies, procedures, dates, requirements, or relationships.
- If the chunk does not provide enough evidence, omit the fact.
- Prefer missing a weak fact over creating an unsupported fact.
- Ignore page numbers, formatting artifacts, headers and isolated
  fragments unless semantically meaningful.

ENTITY RULES

Extract entities useful for enterprise retrieval and reasoning.

Use only these entity types:
ORGANIZATION
POLICY
ROLE
AUTHORITY
LAW_REGULATION
PROCESS
ACTION
REQUIREMENT
DOCUMENT
LOCATION
DATE
CONCEPT
OTHER

Use meaningful canonical names based on the supplied text.

RELATIONSHIP RULES

Extract only relationships directly and explicitly supported by
the supplied chunk.

Every relationship is represented as:

SOURCE --RELATION--> TARGET

The direction MUST match the semantic meaning of the predicate,
not merely the grammatical order of words in the sentence.

Use canonical relationship direction whenever possible.

Examples:

Policy --APPROVED_BY--> Board
Activity --MONITORED_BY--> Board
Employee --REPORTS_TO--> Manager
Procedure --REQUIRES--> Approval
Policy --GOVERNED_BY--> Regulation
Report --SUBMITTED_TO--> Authority
Report --PLACED_BEFORE--> Board
Document --REFERENCES--> Regulation
Entity --PART_OF--> Organization
Person --RESPONSIBLE_FOR--> Process
Whistle-blower --PROTECTED_FROM--> Victimization

INCORRECT:
Board --APPROVED_BY--> Policy

CORRECT:
Policy --APPROVED_BY--> Board

INCORRECT:
Board --MONITORED_BY--> Activity

CORRECT:
Activity --MONITORED_BY--> Board

Do NOT change the meaning of the source text.

For example:

"Impact assessment reports shall be placed before the Board"

means:

Impact Assessment Reports --PLACED_BEFORE--> Board

It does NOT mean:

Impact Assessment Reports --APPROVED_BY--> Board

Do not upgrade weaker statements into stronger facts.

Examples:

"placed before" != "approved by"
"may" != "shall"
"can" != "must"
"references" != "governed by"
"recommends" != "approves"
"participates in" != "responsible for"

The relationship predicate must preserve the strength and meaning
of the original statement.

If no precise relationship can be supported, omit it.

The following are useful canonical predicates when applicable:

APPLIES_TO
APPROVED_BY
MONITORED_BY
GOVERNED_BY
REGULATED_BY
REQUIRES
RESPONSIBLE_FOR
REPORTS_TO
SUBMITTED_TO            
PLACED_BEFORE
PROTECTED_FROM
DEFINED_BY
REFERENCES
PART_OF
FORMULATED_BY
DISPLAYED_ON
ANNEXED_TO
EXCLUDED_FROM
PURSUANT_TO

This vocabulary is NOT closed.

Future documents may concern pipelines, HR, operations, safety,
engineering, finance, procurement, SOPs, tenders or domains not
known today.

When another relationship is explicitly stated, create a concise
predicate that preserves its exact meaning.

Every relationship source and target must appear in the entity
list.

Evidence must directly support the relationship.

Never create a relationship merely because two entities occur
in the same chunk.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=(
                "Extract an evidence-grounded knowledge graph "
                "from the following enterprise document chunk.\n\n"
                "DOCUMENT CHUNK:\n"
                + text
            ),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=KnowledgeExtraction,
            ),
        )

        if not response.text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        extraction = (
            KnowledgeExtraction.model_validate_json(
                response.text
            )
        )

        return self.validate_extraction(
            extraction
        )