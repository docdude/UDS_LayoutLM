"""
NextGen Enterprise API Integration for CRC Audit Workflow
=========================================================

This module will integrate with NextGen USCDI API endpoints for:
1. Pre-filtering structured data (procedures, lab results)
2. Downloading unstructured documents (PDFs) for model triage

API Documentation: https://www.nextgen.com/api/enterprise-uscdi-documentation
Swagger: https://www.nextgen.com/-/media/files/api/enterprise-uscdi-documentation/nge-uscdi-swagger2-0-3-31.json

REQUIRES:
- API Client ID and Secret (contact NextGen Partner Program)
- Enterprise API access credentials

ENDPOINTS USED FOR CRC:
-----------------------
1. GET /persons/lookup - Find patient by MRN
2. GET /persons/{personId}/chart/procedures - Colonoscopy, sigmoidoscopy, CT colonography
3. GET /persons/{personId}/chart/lab/results - FIT, FOBT, FIT-DNA results
4. GET /persons/{personId}/chart/documents - List available documents
5. GET /persons/{personId}/chart/documents/{documentId}/pdf - Download PDF
"""

import os
import requests
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class NextGenAPIConfig:
    """NextGen Enterprise API Configuration"""
    base_url: str = "https://nativeapi.nextgen.com/nge/prod/nge-api/api"
    client_id: str = ""
    client_secret: str = ""
    access_token: Optional[str] = None
    token_expiry: Optional[datetime] = None
    
    @classmethod
    def from_env(cls) -> 'NextGenAPIConfig':
        """Load configuration from environment variables"""
        return cls(
            base_url=os.getenv('NEXTGEN_API_BASE_URL', cls.base_url),
            client_id=os.getenv('NEXTGEN_CLIENT_ID', ''),
            client_secret=os.getenv('NEXTGEN_CLIENT_SECRET', '')
        )


# ============================================================================
# CRC-RELEVANT FILTER CODES
# ============================================================================

# LOINC codes for CRC screening lab tests
CRC_LOINC_CODES = {
    # FOBT (Guaiac)
    '2335-8': {'name': 'FOBT', 'lookback_years': 1},
    '14563-1': {'name': 'FOBT', 'lookback_years': 1},
    '14564-9': {'name': 'FOBT', 'lookback_years': 1},
    
    # FIT (Immunochemical)
    '29771-3': {'name': 'FIT', 'lookback_years': 1},
    '56490-6': {'name': 'FIT', 'lookback_years': 1},
    '57905-2': {'name': 'FIT', 'lookback_years': 1},
    '58453-2': {'name': 'FIT', 'lookback_years': 1},
    '80372-6': {'name': 'FIT', 'lookback_years': 1},
    
    # FIT-DNA (Cologuard)
    '77353-1': {'name': 'FIT-DNA', 'lookback_years': 3},
    '77354-9': {'name': 'FIT-DNA', 'lookback_years': 3},
}

# CPT codes for CRC screening procedures
CRC_CPT_CODES = {
    # Colonoscopy
    '45378': {'name': 'Colonoscopy', 'lookback_years': 10},
    '45380': {'name': 'Colonoscopy with biopsy', 'lookback_years': 10},
    '45381': {'name': 'Colonoscopy with submucosal injection', 'lookback_years': 10},
    '45382': {'name': 'Colonoscopy with control of bleeding', 'lookback_years': 10},
    '45384': {'name': 'Colonoscopy with polypectomy', 'lookback_years': 10},
    '45385': {'name': 'Colonoscopy with snare polypectomy', 'lookback_years': 10},
    '45386': {'name': 'Colonoscopy with dilation', 'lookback_years': 10},
    '45388': {'name': 'Colonoscopy with ablation', 'lookback_years': 10},
    '45390': {'name': 'Colonoscopy with stent placement', 'lookback_years': 10},
    '45391': {'name': 'Colonoscopy with endoscopic ultrasound', 'lookback_years': 10},
    '45392': {'name': 'Colonoscopy with transendoscopic stent placement', 'lookback_years': 10},
    '45393': {'name': 'Colonoscopy with decompression', 'lookback_years': 10},
    '45398': {'name': 'Colonoscopy with band ligation', 'lookback_years': 10},
    
    # Flexible sigmoidoscopy
    '45330': {'name': 'Sigmoidoscopy', 'lookback_years': 5},
    '45331': {'name': 'Sigmoidoscopy with biopsy', 'lookback_years': 5},
    '45332': {'name': 'Sigmoidoscopy with polypectomy', 'lookback_years': 5},
    '45333': {'name': 'Sigmoidoscopy with ablation', 'lookback_years': 5},
    '45334': {'name': 'Sigmoidoscopy with control of bleeding', 'lookback_years': 5},
    '45335': {'name': 'Sigmoidoscopy with submucosal injection', 'lookback_years': 5},
    '45337': {'name': 'Sigmoidoscopy with decompression', 'lookback_years': 5},
    '45338': {'name': 'Sigmoidoscopy with snare polypectomy', 'lookback_years': 5},
    '45340': {'name': 'Sigmoidoscopy with dilation', 'lookback_years': 5},
    '45341': {'name': 'Sigmoidoscopy with ultrasound', 'lookback_years': 5},
    '45342': {'name': 'Sigmoidoscopy with transendoscopic ultrasound', 'lookback_years': 5},
    '45346': {'name': 'Sigmoidoscopy with ablation', 'lookback_years': 5},
    '45347': {'name': 'Sigmoidoscopy with stent placement', 'lookback_years': 5},
    '45349': {'name': 'Sigmoidoscopy with endoscopic mucosal resection', 'lookback_years': 5},
    '45350': {'name': 'Sigmoidoscopy with band ligation', 'lookback_years': 5},
    
    # CT Colonography
    '74261': {'name': 'CT Colonography diagnostic', 'lookback_years': 5},
    '74262': {'name': 'CT Colonography screening', 'lookback_years': 5},
    '74263': {'name': 'CT Colonography screening with contrast', 'lookback_years': 5},
}

# Document type keywords for pre-filtering
CRC_DOCUMENT_KEYWORDS = [
    'colonoscopy', 'endoscopy', 'gastroenterology', 'gi consult',
    'sigmoidoscopy', 'ct colonography', 'virtual colonoscopy',
    'fobt', 'fit', 'fecal immunochemical', 'fecal occult blood',
    'cologuard', 'stool dna', 'colorectal', 'colon cancer screening',
    'lab result', 'pathology', 'procedure report'
]


# ============================================================================
# API CLIENT (Placeholder - requires API tokens)
# ============================================================================
class NextGenAPIClient:
    """
    NextGen Enterprise API Client for CRC Audit Workflow
    
    NOTE: This is a placeholder implementation.
    Actual API calls require valid client credentials.
    """
    
    def __init__(self, config: Optional[NextGenAPIConfig] = None):
        self.config = config or NextGenAPIConfig.from_env()
        self._session = requests.Session()
    
    def authenticate(self) -> bool:
        """
        Authenticate with NextGen API and obtain access token.
        
        See: https://www.nextgen.com/api/EntAPI-GSA-Auth-Docs-Guide.pdf
        """
        # TODO: Implement OAuth2 authentication
        # This typically involves:
        # 1. POST to token endpoint with client_id and client_secret
        # 2. Receive access_token and refresh_token
        # 3. Store token and expiry for subsequent requests
        
        logger.warning("API authentication not implemented - requires credentials")
        return False
    
    def lookup_patient(self, mrn: str) -> Optional[Dict[str, Any]]:
        """
        Find patient by MRN using /persons/lookup
        
        Returns personId needed for subsequent chart queries.
        """
        # GET /persons/lookup?$filter=mrn eq '{mrn}'
        raise NotImplementedError("Requires API credentials")
    
    def get_procedures(
        self, 
        person_id: str, 
        cpt_codes: Optional[List[str]] = None,
        since_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Get patient procedures from structured data.
        
        Endpoint: GET /persons/{personId}/chart/procedures
        
        Use for: Colonoscopy, sigmoidoscopy, CT colonography
        """
        # Can use $filter with CPT codes and date ranges
        # $filter=cptCode eq '45378' and performedDateTime gt dateTime'2016-01-01'
        raise NotImplementedError("Requires API credentials")
    
    def get_lab_results(
        self,
        person_id: str,
        loinc_codes: Optional[List[str]] = None,
        since_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Get patient lab results from structured data.
        
        Endpoint: GET /persons/{personId}/chart/lab/results
        
        Use for: FIT, FOBT, FIT-DNA (Cologuard)
        """
        # Can use $filter with LOINC codes and date ranges
        raise NotImplementedError("Requires API credentials")
    
    def get_documents(
        self,
        person_id: str,
        document_types: Optional[List[str]] = None,
        since_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of patient documents (for unstructured data).
        
        Endpoint: GET /persons/{personId}/chart/documents
        
        Returns document metadata including documentId for PDF download.
        """
        raise NotImplementedError("Requires API credentials")
    
    def download_document_pdf(
        self,
        person_id: str,
        document_id: str,
        output_path: str
    ) -> bool:
        """
        Download document as PDF.
        
        Endpoint: GET /persons/{personId}/chart/documents/{documentId}/pdf
        """
        raise NotImplementedError("Requires API credentials")


# ============================================================================
# CRC AUDIT WORKFLOW
# ============================================================================
class CRCAuditWorkflow:
    """
    Two-stage CRC audit workflow:
    
    Stage 1: Query structured data (procedures, lab results)
             If CRC evidence found in structured data -> DONE
             
    Stage 2: Download unstructured documents (PDFs)
             Run LayoutLMv3 triage model
             Evaluate for CRC evidence
    """
    
    def __init__(self, api_client: NextGenAPIClient):
        self.api = api_client
        self.reporting_year = datetime.now().year
    
    def check_structured_data(self, person_id: str) -> Dict[str, Any]:
        """
        Stage 1: Check structured EHR data for CRC screening evidence.
        
        This is FAST and doesn't require the model.
        """
        result = {
            'has_structured_evidence': False,
            'procedures': [],
            'lab_results': [],
            'best_evidence': None
        }
        
        # Check procedures (colonoscopy, sigmoidoscopy, CT colonography)
        try:
            procedures = self.api.get_procedures(
                person_id,
                cpt_codes=list(CRC_CPT_CODES.keys()),
                since_date=date(self.reporting_year - 10, 1, 1)  # 10 year lookback
            )
            result['procedures'] = procedures
            
            # Check if any procedure is within valid lookback
            for proc in procedures:
                cpt = proc.get('cptCode')
                proc_date = proc.get('performedDateTime')
                if cpt in CRC_CPT_CODES and self._within_lookback(proc_date, CRC_CPT_CODES[cpt]['lookback_years']):
                    result['has_structured_evidence'] = True
                    result['best_evidence'] = proc
                    break
        except NotImplementedError:
            logger.warning("Procedure query not available")
        
        # Check lab results (FIT, FOBT, FIT-DNA)
        if not result['has_structured_evidence']:
            try:
                labs = self.api.get_lab_results(
                    person_id,
                    loinc_codes=list(CRC_LOINC_CODES.keys()),
                    since_date=date(self.reporting_year - 3, 1, 1)  # 3 year max for FIT-DNA
                )
                result['lab_results'] = labs
                
                for lab in labs:
                    loinc = lab.get('loincCode')
                    lab_date = lab.get('collectedDateTime')
                    if loinc in CRC_LOINC_CODES and self._within_lookback(lab_date, CRC_LOINC_CODES[loinc]['lookback_years']):
                        result['has_structured_evidence'] = True
                        result['best_evidence'] = lab
                        break
            except NotImplementedError:
                logger.warning("Lab query not available")
        
        return result
    
    def download_unstructured_documents(self, person_id: str, output_dir: str) -> List[str]:
        """
        Stage 2a: Download unstructured documents for model processing.
        
        Pre-filters by document type keywords to reduce download volume.
        """
        downloaded = []
        
        try:
            documents = self.api.get_documents(
                person_id,
                since_date=date(self.reporting_year - 10, 1, 1)
            )
            
            for doc in documents:
                doc_type = doc.get('documentType', '').lower()
                doc_name = doc.get('name', '').lower()
                
                # Pre-filter by keywords
                is_relevant = any(
                    kw in doc_type or kw in doc_name 
                    for kw in CRC_DOCUMENT_KEYWORDS
                )
                
                if is_relevant:
                    doc_id = doc.get('id')
                    output_path = f"{output_dir}/{person_id}_{doc_id}.pdf"
                    
                    if self.api.download_document_pdf(person_id, doc_id, output_path):
                        downloaded.append(output_path)
            
        except NotImplementedError:
            logger.warning("Document download not available")
        
        return downloaded
    
    def _within_lookback(self, event_date: str, lookback_years: int) -> bool:
        """Check if event date is within lookback period."""
        if not event_date:
            return False
        
        try:
            event = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
            cutoff = datetime(self.reporting_year - lookback_years + 1, 1, 1)
            return event >= cutoff
        except:
            return False


# ============================================================================
# ENTRY POINT FOR TESTING
# ============================================================================
if __name__ == "__main__":
    print("NextGen API Integration for CRC Audit")
    print("=" * 50)
    print()
    print("Status: PLACEHOLDER - API credentials required")
    print()
    print("To use this module:")
    print("1. Apply for NextGen Partner Program")
    print("2. Obtain Client ID and Client Secret")
    print("3. Set environment variables:")
    print("   - NEXTGEN_CLIENT_ID")
    print("   - NEXTGEN_CLIENT_SECRET")
    print()
    print("CRC-relevant codes loaded:")
    print(f"  - {len(CRC_LOINC_CODES)} LOINC codes (lab tests)")
    print(f"  - {len(CRC_CPT_CODES)} CPT codes (procedures)")
    print(f"  - {len(CRC_DOCUMENT_KEYWORDS)} document keywords")
