// ============================================================
// Neo4j Schema Constraints & Indexes
// Run once after Neo4j starts via: docker exec neo4j cypher-shell -u neo4j -p password123 -f /var/lib/neo4j/import/constraints.cypher
// Or: the processor API creates this automatically on startup.
// ============================================================

// --- Uniqueness Constraints ---

// Fantasy LitRPG
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (n:Person) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT location_id IF NOT EXISTS FOR (n:Location) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (n:Skill) REQUIRE n.id IS UNIQUE;

// Game of Thrones
CREATE CONSTRAINT character_id IF NOT EXISTS FOR (n:Character) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT house_id IF NOT EXISTS FOR (n:House) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT got_location_id IF NOT EXISTS FOR (n:Location) REQUIRE n.id IS UNIQUE;

// Legal
CREATE CONSTRAINT paragraph_id IF NOT EXISTS FOR (n:Paragraph) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (n:Rule) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT violation_id IF NOT EXISTS FOR (n:Violation) REQUIRE n.id IS UNIQUE;

// --- Fulltext Search Index ---
// Covers id and description/text_excerpt across all major node types.
// Used by the chatbot for entity matching (replaces slow CONTAINS scan).

CREATE FULLTEXT INDEX node_search IF NOT EXISTS
FOR (n:Person|Character|House|Location|Skill|Class|Monster|Item|Organization|
     Paragraph|Rule|Violation|Prohibition|Requirement|Exception|Fine|Penalty|
     VehicleCategory|PersonRole|RoadType|Permit|Authority|Definition|
     Battle|Alliance|Faction|Title|Weapon|Event)
ON EACH [n.id, n.description, n.text_excerpt, n.paragraph_number, n.paragraph_title];
