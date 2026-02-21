import ipaddress
import pandas as pd
import numpy as np

def fill_age_from_cat(row, age_map):
    """
    Remplit l'âge manquant en utilisant la catégorie d'âge.
    """
    if pd.isna(row['Age']) and row['AgeCategory'] in age_map:
        return age_map[row['AgeCategory']]
    return row['Age']

def extract_ip_features(ip_str):
    """
    Transforme une adresse IP en variables numériques : 
    Version, Premier Octet et Type (Privée/Publique).
    """
    try:
        # Nettoyage des espaces éventuels
        ip_str = str(ip_str).strip()
        ip = ipaddress.ip_address(ip_str)
        version = ip.version
        
        # Extraction du premier octet (différent selon IPv4 ou IPv6)
        if version == 4:
            first_octet = int(ip_str.split('.')[0])
        else:
            first_octet = int(ip_str.split(':')[0], 16)
            
        return version, first_octet, int(ip.is_private)
    except:
        # Valeurs par défaut si l'IP est invalide ou manquante
        return 4, 0, 0