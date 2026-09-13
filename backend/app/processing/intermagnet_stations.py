"""Static roster of INTERMAGNET/IAGA observatory codes, for the "nearest
observatories" base-station estimation feature (see
intermagnet.py:select_nearest_observatories).

This roster (130 stations: IAGA code, station name, country/territory) was
extracted from the string table of a third-party Windows GUI tool a user
shared for downloading INTERMAGNET data ("INTERMAGNET_KMAG.exe") - static
analysis of the binary showed it calls the exact same public BGS GIN web
service this app already uses (imag-data.bgs.ac.uk/GIN_V1/GINServices), so
there was nothing special to integrate with; the useful thing salvaged
from it was this validated code/name/country list plus a reminder to
double check our own request parameters against ones a working client
actually sends (see intermagnet.py's samplesPerDay comment).

Deliberately NOT included here: latitude/longitude. This module only
narrows which stations are worth live-probing for a given target location
(coarse country-centroid distance, see COUNTRY_CENTROIDS below) - the
authoritative coordinates always come from each station's own IAGA-2002
file header ("Geodetic Latitude"/"Geodetic Longitude", parsed by
parse_iaga2002) at the time its data is actually fetched, never from a
hand-maintained table here that could go stale or be wrong.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservatoryRosterEntry:
    iaga_code: str
    name: str
    country: str


OBSERVATORY_ROSTER: list[ObservatoryRosterEntry] = [
    ObservatoryRosterEntry("ABG", "Alibag", "India"),
    ObservatoryRosterEntry("ABK", "Abisko", "Sweden"),
    ObservatoryRosterEntry("AIA", "Vernadsky", "Antarctica"),
    ObservatoryRosterEntry("API", "Apia", "Samoa"),
    ObservatoryRosterEntry("ARS", "Arti", "Russia"),
    ObservatoryRosterEntry("ASC", "Ascension Island", "Ascension Island"),
    ObservatoryRosterEntry("ASP", "Alice Springs", "Australia"),
    ObservatoryRosterEntry("BDV", "Budkov", "Czech Republic"),
    ObservatoryRosterEntry("BEL", "Belsk", "Poland"),
    ObservatoryRosterEntry("BFE", "Brorfelde", "Denmark"),
    ObservatoryRosterEntry("BFO", "Black Forest Observatory", "Germany"),
    ObservatoryRosterEntry("BLC", "Baker Lake", "Canada"),
    ObservatoryRosterEntry("BMT", "Beijing Ming Tombs", "China"),
    ObservatoryRosterEntry("BOU", "Boulder", "USA"),
    ObservatoryRosterEntry("BOX", "Borok", "Russia"),
    ObservatoryRosterEntry("BRD", "Brandon", "Canada"),
    ObservatoryRosterEntry("BRW", "Barrow", "USA"),
    ObservatoryRosterEntry("BSL", "Stennis Space Center", "USA"),
    ObservatoryRosterEntry("CBB", "Cambridge Bay", "Canada"),
    ObservatoryRosterEntry("CKI", "Cocos (Keeling) Islands", "Australia"),
    ObservatoryRosterEntry("CLF", "Chambon la Foret", "France"),
    ObservatoryRosterEntry("CMO", "College", "USA"),
    ObservatoryRosterEntry("CNB", "Canberra", "Australia"),
    ObservatoryRosterEntry("CNH", "Changchun", "China"),
    ObservatoryRosterEntry("CPL", "Choutuppal", "India"),
    ObservatoryRosterEntry("CSY", "Casey", "Antarctica"),
    ObservatoryRosterEntry("CTA", "Charters Towers", "Australia"),
    ObservatoryRosterEntry("CYG", "Cheongyang", "Republic of Korea"),
    ObservatoryRosterEntry("DED", "Deadhorse", "USA"),
    ObservatoryRosterEntry("DLT", "Dalat", "Vietnam"),
    ObservatoryRosterEntry("DOU", "Dourbes", "Belgium"),
    ObservatoryRosterEntry("DUR", "Duronia", "Italy"),
    ObservatoryRosterEntry("EBR", "Ebro", "Spain"),
    ObservatoryRosterEntry("ESK", "Eskdalemuir", "United Kingdom"),
    ObservatoryRosterEntry("EYR", "Eyrewell", "New Zealand"),
    ObservatoryRosterEntry("FCC", "Fort Churchill", "Canada"),
    ObservatoryRosterEntry("FRD", "Fredericksburg", "USA"),
    ObservatoryRosterEntry("FRN", "Fresno", "USA"),
    ObservatoryRosterEntry("FUR", "Furstenfeldenbruck", "Germany"),
    ObservatoryRosterEntry("GAN", "Gan Int. Airport, Addu Atoll", "Maldives"),
    ObservatoryRosterEntry("GCK", "Grocka", "Serbia"),
    ObservatoryRosterEntry("GDH", "Qeqertarsuaq (Godhavn)", "Greenland"),
    ObservatoryRosterEntry("GNA", "Gnangara", "Australia"),
    ObservatoryRosterEntry("GNG", "Gingin", "Australia"),
    ObservatoryRosterEntry("GUA", "Guam", "USA"),
    ObservatoryRosterEntry("GUI", "Guimar-Tenerife", "Spain"),
    ObservatoryRosterEntry("GZH", "Guangzhou", "China"),
    ObservatoryRosterEntry("HAD", "Hartland", "United Kingdom"),
    ObservatoryRosterEntry("HBK", "Hartebeesthoek", "South Africa"),
    ObservatoryRosterEntry("HER", "Hermanus", "South Africa"),
    ObservatoryRosterEntry("HLP", "Hel Observatory", "Poland"),
    ObservatoryRosterEntry("HON", "Honolulu", "USA"),
    ObservatoryRosterEntry("HRB", "Hurbanovo", "Slovakia"),
    ObservatoryRosterEntry("HRN", "Hornsund", "Svalbard"),
    ObservatoryRosterEntry("HUA", "Huancayo", "Peru"),
    ObservatoryRosterEntry("HYB", "Hyderabad", "India"),
    ObservatoryRosterEntry("IPM", "Isla de Pascua", "Chile"),
    ObservatoryRosterEntry("IQA", "Iqaluit", "Canada"),
    ObservatoryRosterEntry("IRT", "Irkutsk (Patrony)", "Russia"),
    ObservatoryRosterEntry("ISK", "Istanbul-Kandili", "Turkey"),
    ObservatoryRosterEntry("IZN", "Iznik", "Turkey"),
    ObservatoryRosterEntry("JAI", "Jaipur", "India"),
    ObservatoryRosterEntry("JCO", "Jim Carrigan Observatory", "USA"),
    ObservatoryRosterEntry("KAK", "Kakioka", "Japan"),
    ObservatoryRosterEntry("KDU", "Kakadu", "Australia"),
    ObservatoryRosterEntry("KEP", "King Edward Point", "South Georgia"),
    ObservatoryRosterEntry("KHB", "Khabarovsk", "Russia"),
    ObservatoryRosterEntry("KIV", "Kiev Dymer", "Ukraine"),
    ObservatoryRosterEntry("KMH", "Keetmanshoop", "Namibia"),
    ObservatoryRosterEntry("KNY", "Kanoya", "Japan"),
    ObservatoryRosterEntry("KOU", "Kourou, Guyana", "France"),
    ObservatoryRosterEntry("LER", "Lerwick", "United Kingdom"),
    ObservatoryRosterEntry("LON", "Lonjsko Polje", "Croatia"),
    ObservatoryRosterEntry("LRM", "Learmonth", "Australia"),
    ObservatoryRosterEntry("LVV", "Lviv", "Ukraine"),
    ObservatoryRosterEntry("LYC", "Lycksele", "Sweden"),
    ObservatoryRosterEntry("MAB", "Manhay", "Belgium"),
    ObservatoryRosterEntry("MAW", "Mawson", "Antarctica"),
    ObservatoryRosterEntry("MCQ", "Macquarie Island", "Australia"),
    ObservatoryRosterEntry("MEA", "Meanook", "Canada"),
    ObservatoryRosterEntry("MGD", "Magadan", "Russia"),
    ObservatoryRosterEntry("MMB", "Memambetsu", "Japan"),
    ObservatoryRosterEntry("MZL", "Manzhouli", "China"),
    ObservatoryRosterEntry("NAQ", "Narsarsuaq", "Greenland"),
    ObservatoryRosterEntry("NCK", "Nagycenk", "Hungary"),
    ObservatoryRosterEntry("NEW", "Newport", "USA"),
    ObservatoryRosterEntry("NGK", "Niemegk", "Germany"),
    ObservatoryRosterEntry("NUR", "Nurmijarvi", "Finland"),
    ObservatoryRosterEntry("NVS", "Novosibirsk", "Russia"),
    ObservatoryRosterEntry("ORC", "Base Orcadas, Laurie Island", "Antarctica"),
    ObservatoryRosterEntry("OTT", "Ottawa", "Canada"),
    ObservatoryRosterEntry("PAF", "Port-aux-Francais", "Kerguelen Islands"),
    ObservatoryRosterEntry("PAG", "Panagjurishte", "Bulgaria"),
    ObservatoryRosterEntry("PEG", "Pedeli", "Greece"),
    ObservatoryRosterEntry("PET", "Paratunka", "Russia"),
    ObservatoryRosterEntry("PHU", "Phutuy", "Vietnam"),
    ObservatoryRosterEntry("PIL", "Pilar", "Argentina"),
    ObservatoryRosterEntry("PPT", "Pamatai, Tahiti", "France"),
    ObservatoryRosterEntry("PST", "Port Stanley", "Falkland Islands"),
    ObservatoryRosterEntry("RES", "Resolute Bay", "Canada"),
    ObservatoryRosterEntry("SBA", "Scott Base", "Antarctica"),
    ObservatoryRosterEntry("SBL", "Sable Island", "Canada"),
    ObservatoryRosterEntry("SFS", "San Fernando", "Spain"),
    ObservatoryRosterEntry("SHE", "Saint Helena", "Saint Helena"),
    ObservatoryRosterEntry("SHU", "Shumagin", "USA"),
    ObservatoryRosterEntry("SIT", "Sitka", "USA"),
    ObservatoryRosterEntry("SJG", "San Juan", "USA"),
    ObservatoryRosterEntry("SOD", "Sodankyla", "Finland"),
    ObservatoryRosterEntry("SPG", "Saint Petersburg", "Russia"),
    ObservatoryRosterEntry("SPT", "San Pablo-Toledo", "Spain"),
    ObservatoryRosterEntry("STJ", "St John's", "Canada"),
    ObservatoryRosterEntry("SUA", "Surlari", "Romania"),
    ObservatoryRosterEntry("TAM", "Tamanrasset", "Algeria"),
    ObservatoryRosterEntry("TDC", "Tristan da Cunha", "Tristan da Cunha"),
    ObservatoryRosterEntry("THL", "Qaanaaq (Thule)", "Greenland"),
    ObservatoryRosterEntry("THY", "Tihany", "Hungary"),
    ObservatoryRosterEntry("TSU", "Tsumeb", "Namibia"),
    ObservatoryRosterEntry("TTB", "Tatuoca", "Brazil"),
    ObservatoryRosterEntry("TUC", "Tucson", "USA"),
    ObservatoryRosterEntry("UPS", "Uppsala", "Sweden"),
    ObservatoryRosterEntry("VAL", "Valentia", "Ireland"),
    ObservatoryRosterEntry("VIC", "Victoria", "Canada"),
    ObservatoryRosterEntry("VNA", "Neumayer Station", "Antarctica"),
    ObservatoryRosterEntry("VOS", "Vostok", "Antarctica"),
    ObservatoryRosterEntry("VSS", "Vassouras", "Brazil"),
    ObservatoryRosterEntry("WIC", "Conrad", "Austria"),
    ObservatoryRosterEntry("WMQ", "Urumqi", "China"),
    ObservatoryRosterEntry("WNG", "Wingst", "Germany"),
    ObservatoryRosterEntry("YAK", "Yakutsk", "Russia"),
    ObservatoryRosterEntry("YKC", "Yellowknife", "Canada"),
]

# Coarse (country/territory centroid, roughly capital-city-level accuracy)
# coordinates - used only to rank the roster above by rough distance to a
# target location so a "nearest observatories" search doesn't need to
# live-probe all ~130 stations, just the most plausible candidates. Never
# used as a station's actual coordinate - see module docstring.
COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "India": (22.0, 79.0),
    "Sweden": (62.0, 15.0),
    "Antarctica": (-75.0, 0.0),
    "Samoa": (-13.8, -171.8),
    "Russia": (61.5, 105.0),
    "Ascension Island": (-7.9, -14.4),
    "Australia": (-25.0, 135.0),
    "Czech Republic": (49.8, 15.5),
    "Poland": (52.0, 19.5),
    "Denmark": (56.0, 10.0),
    "Germany": (51.0, 10.0),
    "Canada": (56.0, -106.0),
    "China": (35.0, 105.0),
    "USA": (39.8, -98.6),
    "France": (46.6, 2.2),
    "Vietnam": (16.0, 108.0),
    "Belgium": (50.6, 4.5),
    "Italy": (42.8, 12.8),
    "Spain": (40.0, -3.7),
    "United Kingdom": (54.0, -2.0),
    "New Zealand": (-41.0, 174.0),
    "Maldives": (3.2, 73.2),
    "Serbia": (44.0, 21.0),
    "Greenland": (72.0, -40.0),
    "South Africa": (-29.0, 24.0),
    "Slovakia": (48.7, 19.5),
    "Svalbard": (78.2, 15.6),
    "Peru": (-9.2, -75.0),
    "Chile": (-35.7, -71.5),
    "Turkey": (39.0, 35.0),
    "Japan": (36.2, 138.3),
    "South Georgia": (-54.4, -36.6),
    "Ukraine": (49.0, 32.0),
    "Namibia": (-22.5, 17.1),
    "Croatia": (45.1, 15.2),
    "Hungary": (47.2, 19.5),
    "Finland": (64.0, 26.0),
    "Kerguelen Islands": (-49.3, 69.5),
    "Bulgaria": (42.7, 25.5),
    "Greece": (39.0, 22.0),
    "Argentina": (-38.4, -63.6),
    "Falkland Islands": (-51.8, -59.5),
    "Saint Helena": (-15.9, -5.7),
    "Romania": (45.9, 25.0),
    "Algeria": (28.0, 3.0),
    "Tristan da Cunha": (-37.1, -12.3),
    "Ireland": (53.4, -8.0),
    "Austria": (47.5, 14.5),
    "Republic of Korea": (36.5, 127.9),
    "Brazil": (-14.2, -51.9),
}
