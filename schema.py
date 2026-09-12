from typing import Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl, EmailStr
from enum import Enum


class StateEnum(str, Enum):
    """Publication status of an entry"""
    public = "public"  # Publicly visible and active
    draft = "draft"  # Not yet published, in editing
    archived = "archived"  # No longer active but kept for records
    suggestion = "suggestion"  # User-submitted, awaiting approval


class Adapter(BaseModel):
    """
    Metadata about the data source and processing adapter.
    Identifies where the data originates and how it's transformed.
    """
    name: str = Field(
        ..., 
        description="Unique identifier for this adapter (e.g., 'potsdam_api', 'berlin_scraper')"
    )
    sourceName: str = Field(
        ..., 
        description="Human-readable name of the data source (e.g., 'City of Potsdam Open Data Portal')"
    )
    sourceUrl: Optional[str] = Field(
        None, 
        description="URL to the data source website or API endpoint"
    )


class Item(BaseModel):
    """
    A community service, location, facility, or event entry in the mittendrin.in directory.
    Represents places, services, and activities available to the community.
    """
    title: str = Field(
        ...,
        description="The official name or title of the service, place, institution, or event. Use the exact, proper name (e.g., 'Stadtbibliothek Potsdam', 'Familienzentrum Mitte')"
    )
    
    titleAddOn: Optional[Dict[str, str]] = Field(
        None,
        description="Translations of the title for non-proper nouns. Keys are ISO 639-1 language codes (e.g., 'en', 'de'). Only needed for generic terms like 'Stadtpark' -> {'en': 'City Park'}. Leave empty for proper nouns like 'Görlitzer Park'"
    )
    
    image: Optional[str] = Field(
        None,
        description="URL to a representative image for this entry (e.g., 'https://example.com/image.jpg')"
    )
    
    state: StateEnum = Field(
        ...,
        description="Publication status: 'public' (visible to all), 'draft' (in progress), 'archived' (no longer active), or 'suggestion' (awaiting approval)"
    )
    
    tags: Optional[List[str]] = Field(
        None,
        description="Lowercase category tags for classification. Format: [a-z0-9_-]. Examples: ['health', 'family', 'education', 'sports']"
    )
    
    primaryTopic: Optional[str] = Field(
        None,
        description="The main topic or primary category for this entry"
    )
    
    brief: Dict[str, str] = Field(
        ...,
        description="Short summary (1-2 sentences) displayed in search results. Keys are language codes (e.g., {'de': 'Kurze Beschreibung', 'en': 'Short description'})"
    )
    
    description: Dict[str, str] = Field(
        ...,
        description="Full description with details about the service/location. Keys are language codes. Should be clear and accessible to general audience"
    )
    
    location_ref: Optional[str] = Field(
        None,
        description="ID reference to another Item whose location data should be used. When set, all location fields (address, coordinates, etc.) are inherited from the referenced item"
    )
    
    location: Optional[str] = Field(
        None,
        description="Name of the location or venue (e.g., 'Sozialstation Torstraße', 'Bürgerhaus Mitte')"
    )
    
    directions: Optional[Dict[str, str]] = Field(
        None,
        description="Instructions for reaching the location. Keys are language codes (e.g., {'de': 'Mit Bus 92 bis Rathaus', 'en': 'Take bus 92 to City Hall'})"
    )
    
    address: Optional[str] = Field(
        None,
        description="Street name and number (e.g., 'Hauptstraße 15', 'Am Markt 3')"
    )
    
    zip: Optional[str] = Field(
        None,
        description="Postal code (e.g., '14467', '10115')"
    )
    
    city: Optional[str] = Field(
        None,
        description="City or municipality name (e.g., 'Potsdam', 'Berlin')"
    )
    
    latitude: Optional[float] = Field(
        None,
        description="GPS latitude in decimal degrees (e.g., 52.531677). Range: -90 to 90"
    )
    
    longitude: Optional[float] = Field(
        None,
        description="GPS longitude in decimal degrees (e.g., 13.381777). Range: -180 to 180"
    )
    
    recurring_event: Optional[str] = Field(
        None,
        description=(
            "JSON-encoded array of recurring schedule rules. Format: [[interval, day, start, end, example_date], ...]. "
            "Example: '[[\"weekly\",\"mon\",\"15:45\",\"16:45\",null]]'. "
            "Fields: [0]=interval ('fixed'|'daily'|'mon-fri'|'weekly'|'bi-weekly'|'three-weekly'|'four-weekly'|'first_of_month'|'second_of_month'|'third_of_month'|'fourth_of_month'|'last_of_month'), "
            "[1]=day ('mon'|'tue'|'wed'|'thu'|'fri'|'sat'|'sun' or null), "
            "[2]=start time ('HH:MM'), "
            "[3]=end time ('HH:MM'), "
            "[4]=example date ('YYYY-MM-DD' or null)"
        )
    )
    
    responsibleInstitution: Optional[str] = Field(
        None,
        description="Name of the organization or institution responsible for this service/location (e.g., 'Stadt Potsdam', 'AWO Bezirksverband')"
    )
    
    sponsors: Optional[str] = Field(
        None,
        description="Name(s) of organizations providing financial support or sponsorship (e.g., 'Bundesministerium für Familie', 'Sparkasse Potsdam')"
    )
    
    website: Optional[str] = Field(
        None,
        description="URL to the official website (e.g., 'https://www.potsdam.de')"
    )
    
    email: Optional[str] = Field(
        None,
        description="Contact email address (e.g., 'info@example.de')"
    )
    
    facebook: Optional[str] = Field(
        None,
        description="Facebook page URL or handle (e.g., 'https://www.facebook.com/potsdam.de')"
    )
    
    whatsapp: Optional[str] = Field(
        None,
        description="WhatsApp contact number or link (e.g., '+491234567890' or WhatsApp link)"
    )
    
    contact: Optional[str] = Field(
        None,
        description="Name of the primary contact person responsible for inquiries (e.g., 'Maria Schmidt', 'Herr Müller')"
    )
    
    phone: Optional[str] = Field(
        None,
        description="Phone number with international prefix (e.g., '+49 331 1234567', '+49 30 98765432')"
    )
    
    mobile: Optional[str] = Field(
        None,
        description="Mobile phone number with international prefix (e.g., '+49 171 1234567')"
    )
    
    editingNote: Optional[str] = Field(
        None,
        description="Internal notes for editors and administrators, not visible to public users"
    )
    
    hours: Optional[Dict[str, str]] = Field(
        None,
        description="Free-text opening or service hours information. Keys are language codes. Use for hours that don't fit recurring_event format (e.g., {'de': 'Mo-Fr 9-17 Uhr, Sa 10-14 Uhr'})"
    )
    
    accessibility: Optional[Dict[str, str]] = Field(
        None,
        description="Accessibility information for people with disabilities. Keys are language codes (e.g., {'de': 'Rollstuhlgerecht, Aufzug vorhanden', 'en': 'Wheelchair accessible, elevator available'})"
    )
    
    charge: Optional[Dict[str, str]] = Field(
        None,
        description="Fee and cost information. Keys are language codes (e.g., {'de': 'Eintritt frei', 'en': 'Free admission'} or {'de': '5€ pro Person', 'en': '5€ per person'})"
    )
    
    venue: Optional[Dict[str, str]] = Field(
        None,
        description="Additional requirements or conditions for participation. Keys are language codes (e.g., {'de': 'Anmeldung erforderlich', 'en': 'Registration required'})"
    )
    
    resubmissionDate: Optional[str] = Field(
        None,
        description="Date when this entry should be reviewed or revalidated. Format: ISO 8601 date string (YYYY-MM-DD)"
    )


class AdapterData(BaseModel):
    """
    Complete dataset from a data adapter including metadata and all items.
    This is the root structure containing source information and the full collection of entries.
    """
    adapter: Adapter = Field(
        ...,
        description="Metadata about the data source and adapter used to fetch and process this data"
    )
    
    lastUpdate: int = Field(
        ...,
        description="Unix timestamp (seconds since epoch) of the last successful data fetch. May differ from when the source data actually changed"
    )
    
    version: Optional[str] = Field(
        None,
        description="Version identifier for this dataset (e.g., '2024-01-15', 'v1.2.3'). Used to track changes and support incremental updates"
    )
    
    itemsRecord: Dict[str, Item] = Field(
        ...,
        description="Dictionary of all items/entries. Keys are unique identifiers (often alphanumeric with hyphens/underscores), values are Item objects. Keys may match the item's internal ID but this is not guaranteed"
    )
