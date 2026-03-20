"""
Supplier HTML scraper for cafe inventory system.

This module parses locally-stored HTML files that simulate supplier websites,
extracting product catalogs and supplier metadata.

Production note: In a real system, these would be HTTP requests to supplier
APIs or websites, with appropriate rate limiting, caching (per robots.txt),
and error handling for network timeouts and 5xx responses. Current
implementation operates on static local files for testing/demo.
"""

import os
import re
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup


def parse_lead_time_days(lead_time_string: str) -> int:
    """
    Extract lead time in days from human-readable text.
    
    Parses strings like:
    - "Lead time: 2-3 business days" → 3
    - "Lead time: 1 day" → 1
    - "2-3 business days" → 3
    
    Extracts the maximum number mentioned to give a conservative estimate.
    Returns 7 days as default if parsing fails (conservative fallback).
    
    Args:
        lead_time_string (str): Lead time description text
    
    Returns:
        int: Lead time in days (1-7, or 7 as default on failure)
    
    Example:
        >>> parse_lead_time_days("Lead time: 2-3 business days")
        3
    """
    try:
        # Find all numbers in the string
        numbers = re.findall(r'\d+', lead_time_string)
        if numbers:
            # Return the maximum number found
            return int(max(numbers))
        return 7
    except (ValueError, AttributeError):
        return 7


def parse_price(price_string: str) -> Optional[Dict[str, Any]]:
    """
    Parse price string into structured components.
    
    Handles Indian rupee notation:
    - "₹850/kg" → {'raw': '₹850/kg', 'amount': 850.0, 'per_unit': 'kg'}
    - "₹72/L" → {'raw': '₹72/L', 'amount': 72.0, 'per_unit': 'L'}
    - "₹4.50/unit" → {'raw': '₹4.50/unit', 'amount': 4.5, 'per_unit': 'unit'}
    
    Returns None (all fields) if parsing fails, allowing graceful degradation
    in downstream code.
    
    Args:
        price_string (str): Price in format "₹[amount]/[unit]"
    
    Returns:
        Optional[Dict[str, Any]]: Dict with keys raw, amount, per_unit,
                                 or None if parsing fails
    
    Example:
        >>> parse_price("₹850/kg")
        {'raw': '₹850/kg', 'amount': 850.0, 'per_unit': 'kg'}
    """
    try:
        # Match: currency symbol, number (int or float), /, unit
        pattern = r'[₹$€]\s*([0-9.]+)\s*/\s*(\w+)'
        match = re.search(pattern, price_string.strip())
        
        if not match:
            return None
        
        amount_str, per_unit = match.groups()
        amount = float(amount_str)
        
        return {
            'raw': price_string.strip(),
            'amount': amount,
            'per_unit': per_unit
        }
    except (ValueError, AttributeError):
        return None


def scrape_supplier_page(filepath: str) -> List[Dict[str, Any]]:
    """
    Scrape product catalog from a single supplier HTML page.
    
    Parses local HTML files following a consistent structure with CSS classes:
    - .supplier-info: header section with company metadata
    - .product: individual product listings
    
    Each returned product dict combines supplier metadata with product details,
    flattened into a single dict for easy downstream processing.
    
    Business guarantees:
    - Returns empty list if file cannot be read or parsed
    - Gracefully handles missing elements (uses BeautifulSoup .get() or defaults)
    - sustainability_score defaults to 50 if missing or invalid
    
    Args:
        filepath (str): Absolute or relative path to HTML file
    
    Returns:
        List[Dict[str, Any]]: List of product dicts, each with keys:
            - supplier_name, location, lead_time_days, tags (list)
            - product_name, price (dict), min_order
            - sustainability_score (int), description
            - source_file (basename)
    
    Example:
        >>> products = scrape_supplier_page('supplier_a.html')
        >>> len(products)
        4
        >>> products[0]['supplier_name']
        'Blue Tokai Wholesale'
    """
    try:
        # Read and parse HTML
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract supplier metadata
        supplier_info = soup.find('div', class_='supplier-info')
        if not supplier_info:
            return []
        
        # Get supplier name from h1
        h1_tag = supplier_info.find('h1')
        supplier_name = h1_tag.get_text(strip=True) if h1_tag else 'Unknown'
        
        # Get location
        location_tag = supplier_info.find('p', class_='location')
        location = location_tag.get_text(strip=True) if location_tag else ''
        
        # Get lead time
        lead_time_tag = supplier_info.find('p', class_='lead-time')
        lead_time_str = lead_time_tag.get_text(strip=True) if lead_time_tag else ''
        lead_time_days = parse_lead_time_days(lead_time_str)
        
        # Get tags
        tags_tag = supplier_info.find('p', class_='tags')
        if tags_tag:
            tags_text = tags_tag.get_text(strip=True)
            # Remove bullet points and split by commas or bullets
            tags_text = tags_text.replace('✓', '').replace('•', ',')
            tags = [t.strip() for t in tags_text.split(',') if t.strip()]
        else:
            tags = []
        
        # Extract products
        products_container = soup.find('div', class_='products')
        if not products_container:
            return []
        
        product_divs = products_container.find_all('div', class_='product')
        results = []
        
        for product_div in product_divs:
            # Get product name
            name_tag = product_div.find('h2', class_='product-name')
            product_name = name_tag.get_text(strip=True) if name_tag else 'Unknown'
            
            # Get price
            price_tag = product_div.find('p', class_='price')
            price_text = price_tag.get_text(strip=True) if price_tag else ''
            price_dict = parse_price(price_text)
            
            # Get min order
            min_order_tag = product_div.find('p', class_='min-order')
            min_order = min_order_tag.get_text(strip=True) if min_order_tag else ''
            
            # Get sustainability score
            score_tag = product_div.find('p', class_='sustainability-score')
            try:
                score_text = score_tag.get_text(strip=True) if score_tag else '50'
                sustainability_score = int(score_text)
            except (ValueError, AttributeError):
                sustainability_score = 50
            
            # Get description
            desc_tag = product_div.find('p', class_='description')
            description = desc_tag.get_text(strip=True) if desc_tag else ''
            
            # Combine supplier metadata with product data
            product_dict = {
                'supplier_name': supplier_name,
                'location': location,
                'lead_time_days': lead_time_days,
                'tags': tags,
                'product_name': product_name,
                'price': price_dict,  # Can be None
                'price_raw': price_text,
                'min_order': min_order,
                'sustainability_score': sustainability_score,
                'description': description,
                'source_file': os.path.basename(filepath)
            }
            
            results.append(product_dict)
        
        return results
    
    except (IOError, OSError) as e:
        # File not found or cannot be read
        print(f"Warning: Could not read {filepath}: {e}")
        return []
    except Exception as e:
        # Unexpected parsing error
        print(f"Warning: Error parsing {filepath}: {e}")
        return []


def scrape_all_suppliers(pages_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scrape all supplier HTML files from a directory.
    
    Automatically locates the mock_pages directory relative to this file
    if pages_dir is not provided.
    
    Args:
        pages_dir (Optional[str]): Directory containing .html files.
                                  Defaults to ./mock_pages/ relative to
                                  this script's location.
    
    Returns:
        List[Dict[str, Any]]: All products from all suppliers, flattened
                             into a single list
    
    Example:
        >>> all_products = scrape_all_suppliers()
        >>> len(all_products)
        15  # Total products across all supplier pages
    """
    # Default to mock_pages folder relative to this file
    if pages_dir is None:
        pages_dir = os.path.join(
            os.path.dirname(__file__),
            'mock_pages'
        )
    
    # Find all .html files
    if not os.path.isdir(pages_dir):
        print(f"Warning: Pages directory not found: {pages_dir}")
        return []
    
    html_files = [
        f for f in os.listdir(pages_dir)
        if f.endswith('.html')
    ]
    
    # Scrape each file
    all_products = []
    for html_file in html_files:
        filepath = os.path.join(pages_dir, html_file)
        products = scrape_supplier_page(filepath)
        all_products.extend(products)
    
    # Print summary
    print(f"Scraped {len(all_products)} products from {len(html_files)} supplier pages")
    
    return all_products
