# NextGen Enterprise API Integration

## Overview

This module provides integration with NextGen Enterprise USCDI API for the CRC audit workflow.

**Status:** 🔴 PLACEHOLDER - Requires API credentials

## Getting API Access

1. Visit [NextGen Partner Program](https://www.nextgen.com/partner-program)
2. Apply for Open Access Partner Program (for NextGen Healthcare Clients)
3. Complete self-registration process
4. Receive Client ID and Client Secret

## API Documentation

- [Enterprise USCDI Documentation](https://www.nextgen.com/api/enterprise-uscdi-documentation)
- [Authentication Guide](https://www.nextgen.com/api/EntAPI-GSA-Auth-Docs-Guide.pdf)
- [Swagger Spec](https://www.nextgen.com/-/media/files/api/enterprise-uscdi-documentation/nge-uscdi-swagger2-0-3-31.json)

## Endpoints Used for CRC Triage

### Stage 1: Structured Data (Fast Pre-Filter)

| Endpoint | Purpose | CRC Use |
|----------|---------|---------|
| `GET /persons/lookup` | Find patient by MRN | Get personId |
| `GET /persons/{personId}/chart/procedures` | Procedure history | Colonoscopy, sigmoidoscopy |
| `GET /persons/{personId}/chart/lab/results` | Lab results | FIT, FOBT, FIT-DNA |

### Stage 2: Unstructured Documents (Model Required)

| Endpoint | Purpose | CRC Use |
|----------|---------|---------|
| `GET /persons/{personId}/chart/documents` | List documents | Find relevant PDFs |
| `GET /persons/{personId}/chart/documents/{id}/pdf` | Download PDF | Model input |

## OData Filter Examples

### Procedures (Colonoscopy within 10 years)
```
GET /persons/{personId}/chart/procedures
  ?$filter=performedDateTime gt dateTime'2016-01-01'
  &$filter=substringof('colonoscopy', name)
```

### Lab Results (FIT within same year)
```
GET /persons/{personId}/chart/lab/results
  ?$filter=collectedDateTime gt dateTime'2025-01-01'
  &$filter=loincCode eq '29771-3'
```

## CRC-Relevant Codes

### LOINC Codes (Lab Tests)

| Code | Test | Lookback |
|------|------|----------|
| 2335-8 | FOBT (guaiac) | Same year |
| 29771-3 | FIT (immunochemical) | Same year |
| 56490-6 | FIT | Same year |
| 77353-1 | FIT-DNA (Cologuard) | 3 years |
| 77354-9 | FIT-DNA | 3 years |

### CPT Codes (Procedures)

| Code | Procedure | Lookback |
|------|-----------|----------|
| 45378-45398 | Colonoscopy | 10 years |
| 45330-45350 | Sigmoidoscopy | 5 years |
| 74261-74263 | CT Colonography | 5 years |

## Environment Variables

```bash
export NEXTGEN_CLIENT_ID="your-client-id"
export NEXTGEN_CLIENT_SECRET="your-client-secret"
export NEXTGEN_API_BASE_URL="https://nativeapi.nextgen.com/nge/prod/nge-api/api"
```

## Usage (When Credentials Available)

```python
from api.nextgen_client import NextGenAPIClient, CRCAuditWorkflow

# Initialize client
client = NextGenAPIClient()
client.authenticate()

# Run two-stage workflow
workflow = CRCAuditWorkflow(client)

# Stage 1: Check structured data first
structured = workflow.check_structured_data(person_id="12345")

if structured['has_structured_evidence']:
    print("CRC evidence found in structured data - no model needed!")
else:
    # Stage 2: Download and process unstructured documents
    pdfs = workflow.download_unstructured_documents(
        person_id="12345",
        output_dir="/tmp/pdfs"
    )
    # Then run LayoutLMv3 model on PDFs
```

## File Structure

```
api/
├── __init__.py
├── nextgen_client.py    # Main API client
└── README.md            # This file
```
