# config/teams.py

TEAM_COLORS = {
    "ANA": {"primary": "#CF4520", "secondary": "#010101"},
    "BOS": {"primary": "#010101", "secondary": "#FFB81C"},
    "BUF": {"primary": "#003087", "secondary": "#FFB81C"},
    "CGY": {"primary": "#C8102E", "secondary": "#F1BE48"},
    "CAR": {"primary": "#010101", "secondary": "#CB102E"},
    "CHI": {"primary": "#C8102E", "secondary": "#010101"},
    "COL": {"primary": "#6F263D", "secondary": "#236192"},
    "CBJ": {"primary": "#041E42", "secondary": "#C8102E"},
    "DAL": {"primary": "#00843D", "secondary": "#010101"},
    "DET": {"primary": "#C8102E", "secondary": "#FFFFFF"},
    "EDM": {"primary": "#00205B", "secondary": "#CF4520"},
    "FLA": {"primary": "#C8102E", "secondary": "#041E42"},
    "LAK": {"primary": "#010101", "secondary": "#A2AAAD"},
    "MIN": {"primary": "#154734", "secondary": "#A6192E"},
    "MTL": {"primary": "#A6192E", "secondary": "#001E62"},
    "NSH": {"primary": "#FFB81C", "secondary": "#041E42"},
    "NJD": {"primary": "#C8102E", "secondary": "#010101"},
    "NYI": {"primary": "#003087", "secondary": "#FC4C02"},
    "NYR": {"primary": "#0032A0", "secondary": "#C8102E"},
    "OTT": {"primary": "#010101", "secondary": "#C8102E"},
    "PHI": {"primary": "#CF4520", "secondary": "#010101"},
    "PIT": {"primary": "#010101", "secondary": "#FFB81C"},
    "STL": {"primary": "#0072CE", "secondary": "#FFB81C"},
    "SJS": {"primary": "#006271", "secondary": "#010101"},
    "SEA": {"primary": "#051C2C", "secondary": "#9CDBD9"},
    "TBL": {"primary": "#00205B", "secondary": "#FFFFFF"},
    "TOR": {"primary": "#00205B", "secondary": "#FFFFFF"},
    "UTA": {"primary": "#010101", "secondary": "#69B3E7"},
    "VAN": {"primary": "#00205B", "secondary": "#00843D"},
    "VGK": {"primary": "#B9975B", "secondary": "#333F48"},
    "WSH": {"primary": "#C8102E", "secondary": "#041E42"},
    "WPG": {"primary": "#041E42", "secondary": "#004C97"},

}

def get_team_colors(abbreviation: str) -> dict:
    """
    Get team colors by abbreviation."""
    return TEAM_COLORS.get(abbreviation.upper(), {
        "primary": "#000000",
        "secondary": "#000000"
    })


# {
#     "name": "Anaheim Ducks",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "a2aaad", "fc4c02", "85714d"]
#     }
#   },
#   {
#     "name": "Arizona Coyotes",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "862633", "ddcba4"]
#     }
#   },
#   {
#     "name": "Boston Bruins",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "ffb81c"]
#     }
#   },
#   {
#     "name": "Buffalo Sabres",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["041e42", "a2aaad", "ffb81c", "c8102e"]
#     }
#   },
#   {
#     "name": "Calgary Flames",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "f1be48", "c8102e"]
#     }
#   },
#   {
#     "name": "Carolina Hurricanes",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "a2aaad", "c8102e"]
#     }
#   },
#   {
#     "name": "Chicago Blackhawks",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "ff671f", "ffd100", "001871", "c8102e", "00843d", "cc8a00"]
#     }
#   },
#   {
#     "name": "Colorado Avalanche",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "#236192", "#a4a9ad", "6f263d"]
#     }
#   },
#   {
#     "name": "Columbus Blue Jackets",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["041e42", "a4a9ad", "c8102e"]
#     }
#   },
#   {
#     "name": "Dallas Stars",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "006341", "8a8d8f"]
#     }
#   },
#   {
#     "name": "Detroit Red Wings",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["c8102e"]
#     }
#   },
#   {
#     "name": "Edmonton Oilers",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["00205b", "cf4520"]
#     }
#   },
#   {
#     "name": "Florida Panthers",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["041e42", "b9975b", "c8102e"]
#     }
#   },
#   {
#     "name": "Los Angeles Kings",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "a2aaad"]
#     }
#   },
#   {
#     "name": "Minnesota Wild",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["154734", "ddcba4", "eaaa00", "a6192e"]
#     }
#   },
#   {
#     "name": "Montreal Canadiens",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["001e62", "a6192e"]
#     }
#   },
#   {
#     "name": "Nashville Predators",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["041e42", "ffb81c"]
#     }
#   },
#   {
#     "name": "New Jersey Devils",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "c8102e"]
#     }
#   },
#   {
#     "name": "New York Islanders",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["003087", "fc4c02"]
#     }
#   },
#   {
#     "name": "New York Rangers",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["0033a0", "c8102e"]
#     }
#   },
#   {
#     "name": "Ottawa Senators",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "c8102e", "c69214"]
#     }
#   },
#   {
#     "name": "Philadelphia Flyers",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "fa4616"]
#     }
#   },
#   {
#     "name": "Pittsburgh Penguins",
#     "league": "nhl",
#     "colors" : {
#       "hex"   :   ["010101", "ffb81c"]
#     }
#   },
#   {
#     "name": "San Jose Sharks",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "e57200", "006272"]
#     }
#   },
#   {
#     "name": "St Louis Blues",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["041e42", "ffb81c", "003087"]
#     }
#   },
#   {
#     "name": "Tampa Bay Lightning",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["00205b"]
#     }
#   },
#   {
#     "name": "Toronto Maple Leafs",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["00205b"]
#     }
#   },
#   {
#     "name": "Vancouver Canucks",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["00205b", "97999b", "041c2c"]
#     }
#   },
#   {
#     "name": "Vegas Golden Knights",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["010101", "b4975a", "333f42"]
#     }
#   },
#   {
#     "name": "Washington Capitals",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["041e42", "a2aaad", "782f40", "a6192e", "53565a"]
#     }
#   },
#   {
#     "name": "Winnipeg Jets",
#     "league": "nhl",
#     "colors": {
#       "hex"   :   ["#041e42", "#c8102e"]
#     }
#   }
