LOAD CSV WITH HEADERS FROM 'http://localhost:11002/project-f28ea7f0-f7b6-4fb5-b817-c1d01ec43864/healthcare.csv' AS row

// Create Patient nodes
MERGE (p:Patient {name: row.Name, age: toInteger(row.Age), gender: row.Gender, bloodType: row.`Blood Type`, medicalCondition: row.`Medical Condition`, dateOfAdmission: date(row.`Date of Admission`), dischargeDate: date(row.`Discharge Date`), medication: row.Medication, testResults: row.`Test Results`, billingAmount: toFloat(row.`Billing Amount`), roomNumber: toInteger(row.`Room Number`), admissionType: row.`Admission Type`})

// Create Condition nodes
MERGE (c:Condition {name: row.`Medical Condition`})

// Create Doctor nodes
MERGE (d:Doctor {name: row.Doctor})

// Create Hospital nodes
MERGE (h:Hospital {name: row.Hospital})

// Create InsuranceProvider nodes
MERGE (i:InsuranceProvider {name: row.`Insurance Provider`})

// Create relationships between nodes
MERGE (p)-[:SUFFERS_FROM]->(c)
MERGE (p)-[:ADMITTED_TO]->(h)
MERGE (p)-[:HAS_DOCTOR]->(d)
MERGE (p)-[:HAS_INSURANCE]->(i)
MERGE (d)-[:WORKS_AT]->(h)
