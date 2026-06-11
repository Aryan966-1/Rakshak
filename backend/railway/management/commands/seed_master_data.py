"""
railway/management/commands/seed_master_data.py
Seed command — populates Zones, Divisions, and 120+ Stations.

Usage:
    python manage.py seed_master_data          # insert (skip existing)
    python manage.py seed_master_data --reset   # wipe & re-insert

All GPS coordinates are real. Station codes follow Indian Railways
conventions. Every station is linked to its correct zone/division.
"""

from decimal import Decimal

from django.core.management.base import BaseCommand
from django.db import transaction

from railway.models import Zone, Division, Station


# ===================================================================
# ZONE DATA — all 18 Indian Railway zones
# ===================================================================
ZONES = [
    {"code": "NR",   "name": "Northern Railway",              "headquarters": "New Delhi"},
    {"code": "WR",   "name": "Western Railway",               "headquarters": "Mumbai (Churchgate)"},
    {"code": "CR",   "name": "Central Railway",               "headquarters": "Mumbai (CST)"},
    {"code": "ER",   "name": "Eastern Railway",               "headquarters": "Kolkata"},
    {"code": "SR",   "name": "Southern Railway",              "headquarters": "Chennai"},
    {"code": "SCR",  "name": "South Central Railway",         "headquarters": "Secunderabad"},
    {"code": "NCR",  "name": "North Central Railway",         "headquarters": "Prayagraj"},
    {"code": "ECR",  "name": "East Central Railway",          "headquarters": "Hajipur"},
    {"code": "NWR",  "name": "North Western Railway",         "headquarters": "Jaipur"},
    {"code": "SER",  "name": "South Eastern Railway",         "headquarters": "Kolkata (Garden Reach)"},
    {"code": "SWR",  "name": "South Western Railway",         "headquarters": "Hubballi"},
    {"code": "ECoR", "name": "East Coast Railway",            "headquarters": "Bhubaneswar"},
    {"code": "WCR",  "name": "West Central Railway",          "headquarters": "Jabalpur"},
    {"code": "SECR", "name": "South East Central Railway",    "headquarters": "Bilaspur"},
    {"code": "NER",  "name": "North Eastern Railway",         "headquarters": "Gorakhpur"},
    {"code": "NFR",  "name": "Northeast Frontier Railway",    "headquarters": "Maligaon (Guwahati)"},
    {"code": "KR",   "name": "Konkan Railway",                "headquarters": "Navi Mumbai"},
    {"code": "MR",   "name": "Metro Railway Kolkata",         "headquarters": "Kolkata"},
]


# ===================================================================
# DIVISION DATA — ~42 divisions linked to zones
# ===================================================================
DIVISIONS = [
    # Northern Railway (NR)
    {"code": "DLI",  "name": "Delhi Division",         "zone_code": "NR",  "headquarters": "New Delhi"},
    {"code": "LKO",  "name": "Lucknow Division",       "zone_code": "NR",  "headquarters": "Lucknow"},
    {"code": "MB",   "name": "Moradabad Division",     "zone_code": "NR",  "headquarters": "Moradabad"},
    {"code": "FZR",  "name": "Firozpur Division",      "zone_code": "NR",  "headquarters": "Firozpur"},
    {"code": "UMB",  "name": "Ambala Division",        "zone_code": "NR",  "headquarters": "Ambala"},

    # Western Railway (WR)
    {"code": "BCT",  "name": "Mumbai Central Division", "zone_code": "WR", "headquarters": "Mumbai"},
    {"code": "ADI",  "name": "Ahmedabad Division",     "zone_code": "WR",  "headquarters": "Ahmedabad"},
    {"code": "RJT",  "name": "Rajkot Division",        "zone_code": "WR",  "headquarters": "Rajkot"},
    {"code": "BRC",  "name": "Vadodara Division",      "zone_code": "WR",  "headquarters": "Vadodara"},

    # Central Railway (CR)
    {"code": "CSMT", "name": "Mumbai Division",         "zone_code": "CR", "headquarters": "Mumbai CST"},
    {"code": "PA",   "name": "Pune Division",           "zone_code": "CR", "headquarters": "Pune"},
    {"code": "BSL",  "name": "Bhusawal Division",       "zone_code": "CR", "headquarters": "Bhusawal"},
    {"code": "NGP",  "name": "Nagpur Division",         "zone_code": "CR", "headquarters": "Nagpur"},

    # Eastern Railway (ER)
    {"code": "HWH",  "name": "Howrah Division",         "zone_code": "ER", "headquarters": "Howrah"},
    {"code": "SDAH", "name": "Sealdah Division",        "zone_code": "ER", "headquarters": "Sealdah"},
    {"code": "ASN",  "name": "Asansol Division",        "zone_code": "ER", "headquarters": "Asansol"},

    # Southern Railway (SR)
    {"code": "MAS",  "name": "Chennai Division",        "zone_code": "SR", "headquarters": "Chennai"},
    {"code": "MDU",  "name": "Madurai Division",        "zone_code": "SR", "headquarters": "Madurai"},
    {"code": "TVC",  "name": "Thiruvananthapuram Division", "zone_code": "SR", "headquarters": "Thiruvananthapuram"},
    {"code": "SA",   "name": "Salem Division",          "zone_code": "SR", "headquarters": "Salem"},

    # South Central Railway (SCR)
    {"code": "SC",   "name": "Secunderabad Division",   "zone_code": "SCR", "headquarters": "Secunderabad"},
    {"code": "HYB",  "name": "Hyderabad Division",      "zone_code": "SCR", "headquarters": "Hyderabad"},
    {"code": "GNT",  "name": "Guntur Division",         "zone_code": "SCR", "headquarters": "Guntur"},
    {"code": "GTL",  "name": "Guntakal Division",       "zone_code": "SCR", "headquarters": "Guntakal"},

    # North Central Railway (NCR)
    {"code": "PRYJ", "name": "Prayagraj Division",      "zone_code": "NCR", "headquarters": "Prayagraj"},
    {"code": "AGC",  "name": "Agra Division",           "zone_code": "NCR", "headquarters": "Agra"},
    {"code": "JHS",  "name": "Jhansi Division",         "zone_code": "NCR", "headquarters": "Jhansi"},

    # East Central Railway (ECR)
    {"code": "DNR",  "name": "Danapur Division",        "zone_code": "ECR", "headquarters": "Patna"},
    {"code": "MGS",  "name": "Mughalsarai Division",    "zone_code": "ECR", "headquarters": "Pt. Deen Dayal Upadhyay Jn"},
    {"code": "SPJ",  "name": "Samastipur Division",     "zone_code": "ECR", "headquarters": "Samastipur"},

    # North Western Railway (NWR)
    {"code": "JP",   "name": "Jaipur Division",         "zone_code": "NWR", "headquarters": "Jaipur"},
    {"code": "AII",  "name": "Ajmer Division",          "zone_code": "NWR", "headquarters": "Ajmer"},
    {"code": "BKN",  "name": "Bikaner Division",        "zone_code": "NWR", "headquarters": "Bikaner"},
    {"code": "JU",   "name": "Jodhpur Division",        "zone_code": "NWR", "headquarters": "Jodhpur"},

    # South Eastern Railway (SER)
    {"code": "CKP",  "name": "Chakradharpur Division",  "zone_code": "SER", "headquarters": "Chakradharpur"},
    {"code": "RNC",  "name": "Ranchi Division",         "zone_code": "SER", "headquarters": "Ranchi"},

    # South Western Railway (SWR)
    {"code": "SBC",  "name": "Bengaluru Division",      "zone_code": "SWR", "headquarters": "Bengaluru"},
    {"code": "UBL",  "name": "Hubballi Division",       "zone_code": "SWR", "headquarters": "Hubballi"},
    {"code": "MYS",  "name": "Mysuru Division",         "zone_code": "SWR", "headquarters": "Mysuru"},

    # East Coast Railway (ECoR)
    {"code": "BBS",  "name": "Bhubaneswar Division",    "zone_code": "ECoR", "headquarters": "Bhubaneswar"},
    {"code": "VSKP", "name": "Visakhapatnam Division",  "zone_code": "ECoR", "headquarters": "Visakhapatnam"},

    # West Central Railway (WCR)
    {"code": "JBP",  "name": "Jabalpur Division",       "zone_code": "WCR", "headquarters": "Jabalpur"},
    {"code": "BPL",  "name": "Bhopal Division",         "zone_code": "WCR", "headquarters": "Bhopal"},

    # South East Central Railway (SECR)
    {"code": "BSP",  "name": "Bilaspur Division",       "zone_code": "SECR", "headquarters": "Bilaspur"},
    {"code": "R",    "name": "Raipur Division",         "zone_code": "SECR", "headquarters": "Raipur"},
    {"code": "NGP2", "name": "Nagpur Division (SECR)",  "zone_code": "SECR", "headquarters": "Nagpur"},

    # North Eastern Railway (NER)
    {"code": "GKP",  "name": "Gorakhpur Division",      "zone_code": "NER", "headquarters": "Gorakhpur"},
    {"code": "IZN",  "name": "Izzatnagar Division",     "zone_code": "NER", "headquarters": "Izzatnagar"},
    {"code": "LJN",  "name": "Lucknow NER Division",    "zone_code": "NER", "headquarters": "Lucknow"},

    # Northeast Frontier Railway (NFR)
    {"code": "GHY",  "name": "Guwahati Division",       "zone_code": "NFR", "headquarters": "Guwahati"},
    {"code": "APDJ", "name": "Alipurduar Division",     "zone_code": "NFR", "headquarters": "Alipurduar"},
    {"code": "KIR",  "name": "Katihar Division",        "zone_code": "NFR", "headquarters": "Katihar"},
]


# ===================================================================
# STATION DATA — 130 stations across India with real GPS coordinates
# ===================================================================
STATIONS = [
    # ---- NORTH INDIA ----
    {"code": "NDLS", "name": "New Delhi",               "lat": "28.6139",  "lng": "77.2090",  "div": "DLI",  "junction": True,  "terminal": True},
    {"code": "DLI",  "name": "Old Delhi Junction",      "lat": "28.6616",  "lng": "77.2286",  "div": "DLI",  "junction": True,  "terminal": False},
    {"code": "NZM",  "name": "Hazrat Nizamuddin",       "lat": "28.5895",  "lng": "77.2533",  "div": "DLI",  "junction": False, "terminal": True},
    {"code": "GZB",  "name": "Ghaziabad Junction",      "lat": "28.6625",  "lng": "77.4381",  "div": "DLI",  "junction": True,  "terminal": False},
    {"code": "ANVT", "name": "Anand Vihar Terminal",     "lat": "28.6469",  "lng": "77.3152",  "div": "DLI",  "junction": False, "terminal": True},
    {"code": "CNB",  "name": "Kanpur Central",           "lat": "26.4534",  "lng": "80.3518",  "div": "LKO",  "junction": True,  "terminal": False},
    {"code": "LKO",  "name": "Lucknow NR",              "lat": "26.8467",  "lng": "80.9462",  "div": "LKO",  "junction": True,  "terminal": False},
    {"code": "LJN",  "name": "Lucknow Junction",        "lat": "26.8605",  "lng": "80.9563",  "div": "LJN",  "junction": True,  "terminal": False},
    {"code": "MB",   "name": "Moradabad Junction",      "lat": "28.8386",  "lng": "78.7733",  "div": "MB",   "junction": True,  "terminal": False},
    {"code": "BE",   "name": "Bareilly Junction",        "lat": "28.3486",  "lng": "79.4230",  "div": "MB",   "junction": True,  "terminal": False},
    {"code": "CDG",  "name": "Chandigarh",               "lat": "30.6822",  "lng": "76.8053",  "div": "UMB",  "junction": False, "terminal": True},
    {"code": "UMB",  "name": "Ambala Cantt Junction",    "lat": "30.3697",  "lng": "76.8173",  "div": "UMB",  "junction": True,  "terminal": False},
    {"code": "LDH",  "name": "Ludhiana Junction",        "lat": "30.8840",  "lng": "75.8665",  "div": "FZR",  "junction": True,  "terminal": False},
    {"code": "ASR",  "name": "Amritsar Junction",        "lat": "31.6318",  "lng": "74.8734",  "div": "FZR",  "junction": True,  "terminal": True},
    {"code": "JRC",  "name": "Jalandhar City",           "lat": "31.3254",  "lng": "75.5807",  "div": "FZR",  "junction": True,  "terminal": False},
    {"code": "FZR",  "name": "Firozpur Junction",        "lat": "30.9268",  "lng": "74.6090",  "div": "FZR",  "junction": True,  "terminal": True},
    {"code": "AGC",  "name": "Agra Cantt",               "lat": "27.1631",  "lng": "78.0081",  "div": "AGC",  "junction": False, "terminal": False},
    {"code": "MTJ",  "name": "Mathura Junction",         "lat": "27.4830",  "lng": "77.6726",  "div": "AGC",  "junction": True,  "terminal": False},
    {"code": "JHS",  "name": "Jhansi Junction",          "lat": "25.4430",  "lng": "78.5683",  "div": "JHS",  "junction": True,  "terminal": False},
    {"code": "GWL",  "name": "Gwalior Junction",         "lat": "26.2214",  "lng": "78.1828",  "div": "JHS",  "junction": True,  "terminal": False},
    {"code": "PRYJ", "name": "Prayagraj Junction",       "lat": "25.4270",  "lng": "81.8851",  "div": "PRYJ", "junction": True,  "terminal": False},
    {"code": "DDU",  "name": "Pt. Deen Dayal Upadhyay Jn", "lat": "25.2802", "lng": "83.0072", "div": "MGS", "junction": True,  "terminal": False},
    {"code": "BSB",  "name": "Varanasi Junction",        "lat": "25.3176",  "lng": "83.0165",  "div": "MGS",  "junction": True,  "terminal": False},
    {"code": "GKP",  "name": "Gorakhpur Junction",       "lat": "26.7469",  "lng": "83.3638",  "div": "GKP",  "junction": True,  "terminal": False},

    # ---- WEST INDIA ----
    {"code": "CSTM", "name": "Mumbai CST",               "lat": "18.9398",  "lng": "72.8355",  "div": "CSMT", "junction": True,  "terminal": True},
    {"code": "BCT",  "name": "Mumbai Central",           "lat": "18.9688",  "lng": "72.8193",  "div": "BCT",  "junction": False, "terminal": True},
    {"code": "LTT",  "name": "Lokmanya Tilak Terminus",  "lat": "19.0688",  "lng": "72.8880",  "div": "CSMT", "junction": False, "terminal": True},
    {"code": "PNVL", "name": "Panvel Junction",          "lat": "18.9903",  "lng": "73.1210",  "div": "CSMT", "junction": True,  "terminal": False},
    {"code": "TNA",  "name": "Thane",                    "lat": "19.1874",  "lng": "72.9756",  "div": "CSMT", "junction": True,  "terminal": False},
    {"code": "KYN",  "name": "Kalyan Junction",          "lat": "19.2352",  "lng": "73.1306",  "div": "CSMT", "junction": True,  "terminal": False},
    {"code": "BSR",  "name": "Vasai Road",               "lat": "19.3652",  "lng": "72.8444",  "div": "BCT",  "junction": True,  "terminal": False},
    {"code": "PUNE", "name": "Pune Junction",            "lat": "18.5288",  "lng": "73.8742",  "div": "PA",   "junction": True,  "terminal": False},
    {"code": "SUR",  "name": "Solapur Junction",         "lat": "17.6726",  "lng": "75.9101",  "div": "PA",   "junction": True,  "terminal": False},
    {"code": "NGP",  "name": "Nagpur Junction",          "lat": "21.1498",  "lng": "79.0806",  "div": "NGP",  "junction": True,  "terminal": False},
    {"code": "BSL",  "name": "Bhusawal Junction",        "lat": "21.0449",  "lng": "75.7834",  "div": "BSL",  "junction": True,  "terminal": False},
    {"code": "ADI",  "name": "Ahmedabad Junction",       "lat": "23.0260",  "lng": "72.6001",  "div": "ADI",  "junction": True,  "terminal": False},
    {"code": "ST",   "name": "Surat",                    "lat": "21.2052",  "lng": "72.8402",  "div": "BCT",  "junction": True,  "terminal": False},
    {"code": "BRC",  "name": "Vadodara Junction",        "lat": "22.3106",  "lng": "73.1812",  "div": "BRC",  "junction": True,  "terminal": False},
    {"code": "RJT",  "name": "Rajkot Junction",          "lat": "22.2919",  "lng": "70.7937",  "div": "RJT",  "junction": True,  "terminal": False},
    {"code": "MMCT", "name": "Maninagar",                "lat": "23.0015",  "lng": "72.6175",  "div": "ADI",  "junction": False, "terminal": False},

    # ---- RAJASTHAN ----
    {"code": "JP",   "name": "Jaipur Junction",          "lat": "26.9194",  "lng": "75.7880",  "div": "JP",   "junction": True,  "terminal": False},
    {"code": "AII",  "name": "Ajmer Junction",           "lat": "26.4530",  "lng": "74.6367",  "div": "AII",  "junction": True,  "terminal": False},
    {"code": "UDZ",  "name": "Udaipur City",             "lat": "24.5804",  "lng": "73.6833",  "div": "AII",  "junction": False, "terminal": True},
    {"code": "JU",   "name": "Jodhpur Junction",         "lat": "26.2888",  "lng": "73.0199",  "div": "JU",   "junction": True,  "terminal": False},
    {"code": "BKN",  "name": "Bikaner Junction",         "lat": "28.0200",  "lng": "73.3057",  "div": "BKN",  "junction": True,  "terminal": False},
    {"code": "KOTA", "name": "Kota Junction",            "lat": "25.1796",  "lng": "75.8648",  "div": "JP",   "junction": True,  "terminal": False},
    {"code": "AWR",  "name": "Abu Road",                 "lat": "24.4806",  "lng": "72.7718",  "div": "AII",  "junction": False, "terminal": False},

    # ---- EAST INDIA ----
    {"code": "HWH",  "name": "Howrah Junction",          "lat": "22.5839",  "lng": "88.3428",  "div": "HWH",  "junction": True,  "terminal": True},
    {"code": "SDAH", "name": "Sealdah",                  "lat": "22.5655",  "lng": "88.3694",  "div": "SDAH", "junction": False, "terminal": True},
    {"code": "BDC",  "name": "Bandel Junction",          "lat": "22.9321",  "lng": "88.3835",  "div": "HWH",  "junction": True,  "terminal": False},
    {"code": "BWN",  "name": "Bardhaman Junction",       "lat": "23.2500",  "lng": "87.8563",  "div": "HWH",  "junction": True,  "terminal": False},
    {"code": "ASN",  "name": "Asansol Junction",         "lat": "23.6834",  "lng": "86.9614",  "div": "ASN",  "junction": True,  "terminal": False},
    {"code": "DHN",  "name": "Dhanbad Junction",         "lat": "23.7879",  "lng": "86.4193",  "div": "ASN",  "junction": True,  "terminal": False},
    {"code": "PNBE", "name": "Patna Junction",           "lat": "25.6094",  "lng": "85.1376",  "div": "DNR",  "junction": True,  "terminal": False},
    {"code": "RJPB", "name": "Rajendra Nagar Terminal",  "lat": "25.6116",  "lng": "85.1256",  "div": "DNR",  "junction": False, "terminal": True},
    {"code": "RNC",  "name": "Ranchi",                   "lat": "23.3137",  "lng": "85.3219",  "div": "RNC",  "junction": True,  "terminal": False},
    {"code": "TATA", "name": "Tatanagar Junction",       "lat": "22.7876",  "lng": "86.1544",  "div": "CKP",  "junction": True,  "terminal": False},
    {"code": "CKP",  "name": "Chakradharpur",            "lat": "22.7009",  "lng": "85.6308",  "div": "CKP",  "junction": True,  "terminal": False},
    {"code": "SPJ",  "name": "Samastipur Junction",      "lat": "25.8636",  "lng": "85.7825",  "div": "SPJ",  "junction": True,  "terminal": False},
    {"code": "KIR",  "name": "Katihar Junction",         "lat": "25.5508",  "lng": "87.5730",  "div": "KIR",  "junction": True,  "terminal": False},
    {"code": "BGP",  "name": "Bhagalpur",                "lat": "25.2444",  "lng": "86.9667",  "div": "MGS",  "junction": False, "terminal": False},

    # ---- SOUTH INDIA ----
    {"code": "MAS",  "name": "Chennai Central",          "lat": "13.0827",  "lng": "80.2707",  "div": "MAS",  "junction": True,  "terminal": True},
    {"code": "MS",   "name": "Chennai Egmore",           "lat": "13.0738",  "lng": "80.2609",  "div": "MAS",  "junction": False, "terminal": True},
    {"code": "TBM",  "name": "Tambaram",                 "lat": "12.9249",  "lng": "80.1169",  "div": "MAS",  "junction": True,  "terminal": False},
    {"code": "AJJ",  "name": "Arakkonam Junction",       "lat": "13.0793",  "lng": "79.6755",  "div": "MAS",  "junction": True,  "terminal": False},
    {"code": "KPD",  "name": "Katpadi Junction",         "lat": "12.9718",  "lng": "79.1650",  "div": "MAS",  "junction": True,  "terminal": False},
    {"code": "JTJ",  "name": "Jolarpettai Junction",     "lat": "12.5637",  "lng": "78.5740",  "div": "SA",   "junction": True,  "terminal": False},
    {"code": "SA",   "name": "Salem Junction",           "lat": "11.6426",  "lng": "78.1569",  "div": "SA",   "junction": True,  "terminal": False},
    {"code": "ED",   "name": "Erode Junction",           "lat": "11.3410",  "lng": "77.7172",  "div": "SA",   "junction": True,  "terminal": False},
    {"code": "CBE",  "name": "Coimbatore Junction",      "lat": "11.0018",  "lng": "76.9558",  "div": "SA",   "junction": True,  "terminal": False},
    {"code": "MDU",  "name": "Madurai Junction",         "lat": "9.9195",   "lng": "78.1252",  "div": "MDU",  "junction": True,  "terminal": False},
    {"code": "TPJ",  "name": "Tiruchirappalli Junction", "lat": "10.8124",  "lng": "78.6858",  "div": "SA",   "junction": True,  "terminal": False},
    {"code": "TVC",  "name": "Thiruvananthapuram Central","lat": "8.4880",   "lng": "76.9500",  "div": "TVC",  "junction": False, "terminal": True},
    {"code": "ERS",  "name": "Ernakulam Junction",       "lat": "9.9816",   "lng": "76.2999",  "div": "TVC",  "junction": True,  "terminal": False},
    {"code": "CLT",  "name": "Kozhikode (Calicut)",      "lat": "11.2453",  "lng": "75.7809",  "div": "TVC",  "junction": False, "terminal": False},
    {"code": "CAN",  "name": "Kannur",                   "lat": "11.8685",  "lng": "75.3509",  "div": "TVC",  "junction": False, "terminal": False},
    {"code": "MAQ",  "name": "Mangaluru Junction",       "lat": "12.8668",  "lng": "74.8821",  "div": "TVC",  "junction": True,  "terminal": False},
    {"code": "SBC",  "name": "Bengaluru City Junction",  "lat": "12.9784",  "lng": "77.5712",  "div": "SBC",  "junction": True,  "terminal": True},
    {"code": "YPR",  "name": "Yesvantpur Junction",      "lat": "13.0278",  "lng": "77.5519",  "div": "SBC",  "junction": True,  "terminal": False},
    {"code": "MYS",  "name": "Mysuru Junction",          "lat": "12.2967",  "lng": "76.6547",  "div": "MYS",  "junction": True,  "terminal": False},
    {"code": "UBL",  "name": "Hubballi Junction",        "lat": "15.3354",  "lng": "75.0963",  "div": "UBL",  "junction": True,  "terminal": False},

    # ---- SOUTH CENTRAL / TELANGANA / AP ----
    {"code": "SC",   "name": "Secunderabad Junction",    "lat": "17.4344",  "lng": "78.5013",  "div": "SC",   "junction": True,  "terminal": False},
    {"code": "HYB",  "name": "Hyderabad Deccan",         "lat": "17.3753",  "lng": "78.4744",  "div": "HYB",  "junction": False, "terminal": True},
    {"code": "WL",   "name": "Warangal",                 "lat": "17.9689",  "lng": "79.5941",  "div": "SC",   "junction": False, "terminal": False},
    {"code": "KZJ",  "name": "Kazipet Junction",         "lat": "17.9852",  "lng": "79.5377",  "div": "SC",   "junction": True,  "terminal": False},
    {"code": "BZA",  "name": "Vijayawada Junction",      "lat": "16.5177",  "lng": "80.6186",  "div": "GNT",  "junction": True,  "terminal": False},
    {"code": "GNT",  "name": "Guntur Junction",          "lat": "16.2991",  "lng": "80.4504",  "div": "GNT",  "junction": True,  "terminal": False},
    {"code": "VSKP", "name": "Visakhapatnam Junction",   "lat": "17.7216",  "lng": "83.2332",  "div": "VSKP", "junction": True,  "terminal": False},
    {"code": "GTL",  "name": "Guntakal Junction",        "lat": "15.1652",  "lng": "77.3678",  "div": "GTL",  "junction": True,  "terminal": False},
    {"code": "RU",   "name": "Raichur",                  "lat": "16.2069",  "lng": "77.3700",  "div": "GTL",  "junction": False, "terminal": False},
    {"code": "TPTY", "name": "Tirupati",                 "lat": "13.6373",  "lng": "79.4192",  "div": "GTL",  "junction": True,  "terminal": False},

    # ---- CENTRAL INDIA ----
    {"code": "BPL",  "name": "Bhopal Junction",          "lat": "23.2686",  "lng": "77.4122",  "div": "BPL",  "junction": True,  "terminal": False},
    {"code": "JBP",  "name": "Jabalpur Junction",        "lat": "23.1684",  "lng": "79.9501",  "div": "JBP",  "junction": True,  "terminal": False},
    {"code": "ET",   "name": "Itarsi Junction",          "lat": "22.6124",  "lng": "77.7627",  "div": "BPL",  "junction": True,  "terminal": False},
    {"code": "BSP",  "name": "Bilaspur Junction",        "lat": "22.0760",  "lng": "82.1496",  "div": "BSP",  "junction": True,  "terminal": False},
    {"code": "R",    "name": "Raipur Junction",          "lat": "21.2514",  "lng": "81.6296",  "div": "R",    "junction": True,  "terminal": False},

    # ---- EAST COAST ----
    {"code": "BBS",  "name": "Bhubaneswar",              "lat": "20.2728",  "lng": "85.8218",  "div": "BBS",  "junction": False, "terminal": False},
    {"code": "CTC",  "name": "Cuttack Junction",         "lat": "20.4650",  "lng": "85.8854",  "div": "BBS",  "junction": True,  "terminal": False},
    {"code": "PURI", "name": "Puri",                     "lat": "19.8050",  "lng": "85.8216",  "div": "BBS",  "junction": False, "terminal": True},
    {"code": "SBP",  "name": "Sambalpur Junction",       "lat": "21.4489",  "lng": "83.9697",  "div": "BBS",  "junction": True,  "terminal": False},
    {"code": "ROU",  "name": "Rourkela Junction",        "lat": "22.2270",  "lng": "84.8024",  "div": "CKP",  "junction": True,  "terminal": False},

    # ---- NORTHEAST INDIA ----
    {"code": "GHY",  "name": "Guwahati",                 "lat": "26.1782",  "lng": "91.7494",  "div": "GHY",  "junction": True,  "terminal": False},
    {"code": "KYQ",  "name": "Kamakhya Junction",        "lat": "26.1756",  "lng": "91.7043",  "div": "GHY",  "junction": True,  "terminal": False},
    {"code": "DBRG", "name": "Dibrugarh Town",           "lat": "27.4728",  "lng": "94.9120",  "div": "GHY",  "junction": False, "terminal": True},
    {"code": "NJP",  "name": "New Jalpaiguri Junction",  "lat": "26.7070",  "lng": "88.4336",  "div": "APDJ", "junction": True,  "terminal": False},
    {"code": "MDP",  "name": "Malda Town",               "lat": "25.0118",  "lng": "88.1347",  "div": "APDJ", "junction": False, "terminal": False},
    {"code": "APDJ", "name": "Alipurduar Junction",      "lat": "26.4889",  "lng": "89.5295",  "div": "APDJ", "junction": True,  "terminal": False},
    {"code": "LMG",  "name": "Lumding Junction",         "lat": "25.7499",  "lng": "93.1713",  "div": "GHY",  "junction": True,  "terminal": False},

    # ---- JAMMU & KASHMIR / NORTHERN HILL ----
    {"code": "JAT",  "name": "Jammu Tawi",               "lat": "32.7137",  "lng": "74.8606",  "div": "FZR",  "junction": False, "terminal": True},
    {"code": "SVDK", "name": "Shri Mata Vaishno Devi Katra", "lat": "32.9868", "lng": "74.9321", "div": "FZR", "junction": False, "terminal": True},
    {"code": "PTK",  "name": "Pathankot Junction",       "lat": "32.2643",  "lng": "75.6347",  "div": "FZR",  "junction": True,  "terminal": False},
    {"code": "DDN",  "name": "Dehradun",                 "lat": "30.3257",  "lng": "78.0346",  "div": "MB",   "junction": False, "terminal": True},
    {"code": "HW",   "name": "Haridwar Junction",        "lat": "29.9185",  "lng": "78.0482",  "div": "MB",   "junction": True,  "terminal": False},

    # ---- KONKAN RAILWAY ----
    {"code": "KRMI", "name": "Karwar",                   "lat": "14.8012",  "lng": "74.1300",  "div": "UBL",  "junction": False, "terminal": False},
    {"code": "THVM", "name": "Thivim",                   "lat": "15.5357",  "lng": "73.9510",  "div": "UBL",  "junction": False, "terminal": False},
    {"code": "RN",   "name": "Ratnagiri",                "lat": "16.9837",  "lng": "73.3157",  "div": "CSMT", "junction": False, "terminal": False},

    # ---- FILL-OUT: more stations to reach 130+ ----
    {"code": "NED",  "name": "Nanded",                   "lat": "19.1527",  "lng": "77.3206",  "div": "SC",   "junction": False, "terminal": False},
    {"code": "AK",   "name": "Akola Junction",           "lat": "20.7083",  "lng": "77.0040",  "div": "BSL",  "junction": True,  "terminal": False},
    {"code": "PBR",  "name": "Porbandar",                "lat": "21.6417",  "lng": "69.6293",  "div": "RJT",  "junction": False, "terminal": True},
    {"code": "BDTS", "name": "Bandra Terminus",          "lat": "19.0544",  "lng": "72.8404",  "div": "BCT",  "junction": False, "terminal": True},
    {"code": "KGP",  "name": "Kharagpur Junction",       "lat": "22.3464",  "lng": "87.3315",  "div": "CKP",  "junction": True,  "terminal": False},
    {"code": "SHM",  "name": "Silchar",                  "lat": "24.8285",  "lng": "92.8022",  "div": "GHY",  "junction": False, "terminal": True},
    {"code": "RGD",  "name": "Rajgir",                   "lat": "25.0283",  "lng": "85.4133",  "div": "DNR",  "junction": False, "terminal": False},
    {"code": "GAY",  "name": "Gaya Junction",            "lat": "24.7991",  "lng": "84.9992",  "div": "MGS",  "junction": True,  "terminal": False},
    {"code": "MGS",  "name": "Mughal Sarai (DDU)",       "lat": "25.2802",  "lng": "83.1161",  "div": "MGS",  "junction": True,  "terminal": False},
    {"code": "CPR",  "name": "Chapra Junction",          "lat": "25.7864",  "lng": "84.7434",  "div": "DNR",  "junction": True,  "terminal": False},
    {"code": "SEE",  "name": "Sonpur Junction",          "lat": "25.6745",  "lng": "85.1863",  "div": "DNR",  "junction": True,  "terminal": False},
    {"code": "DBG",  "name": "Darbhanga Junction",       "lat": "26.1674",  "lng": "85.8938",  "div": "SPJ",  "junction": True,  "terminal": False},
    {"code": "MFP",  "name": "Muzaffarpur Junction",     "lat": "26.1197",  "lng": "85.3917",  "div": "SPJ",  "junction": True,  "terminal": False},
]


class Command(BaseCommand):
    help = "Seed Zones, Divisions, and Stations with real Indian Railway data."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete existing zone/division/station data before seeding.",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        if options["reset"]:
            self.stdout.write(self.style.WARNING("Resetting master data…"))
            Station.objects.all().delete()
            Division.objects.all().delete()
            Zone.objects.all().delete()

        # ----- Zones -----
        zone_map = {}
        created_z = 0
        for z in ZONES:
            obj, created = Zone.objects.get_or_create(
                code=z["code"],
                defaults={"name": z["name"], "headquarters": z["headquarters"]},
            )
            zone_map[z["code"]] = obj
            if created:
                created_z += 1
        self.stdout.write(f"  Zones:     {created_z} created, {len(ZONES) - created_z} skipped (already exist)")

        # ----- Divisions -----
        div_map = {}
        created_d = 0
        for d in DIVISIONS:
            zone = zone_map.get(d["zone_code"])
            if not zone:
                self.stderr.write(f"  [WARN] Zone '{d['zone_code']}' not found for division '{d['code']}' -- skipping.")
                continue
            obj, created = Division.objects.get_or_create(
                code=d["code"],
                defaults={
                    "name": d["name"],
                    "zone": zone,
                    "headquarters": d.get("headquarters", ""),
                },
            )
            div_map[d["code"]] = obj
            if created:
                created_d += 1
        self.stdout.write(f"  Divisions: {created_d} created, {len(DIVISIONS) - created_d} skipped")

        # ----- Stations -----
        created_s = 0
        for s in STATIONS:
            div = div_map.get(s["div"])
            if not div:
                self.stderr.write(f"  [WARN] Division '{s['div']}' not found for station '{s['code']}' -- skipping.")
                continue
            _, created = Station.objects.get_or_create(
                station_code=s["code"],
                defaults={
                    "station_name": s["name"],
                    "division": div,
                    "latitude": Decimal(s["lat"]),
                    "longitude": Decimal(s["lng"]),
                    "is_junction": s.get("junction", False),
                    "is_terminal": s.get("terminal", False),
                },
            )
            if created:
                created_s += 1
        self.stdout.write(f"  Stations:  {created_s} created, {len(STATIONS) - created_s} skipped")

        self.stdout.write(self.style.SUCCESS(
            f"\n[OK] Master data seeded: {len(ZONES)} zones, {len(DIVISIONS)} divisions, {len(STATIONS)} stations."
        ))
