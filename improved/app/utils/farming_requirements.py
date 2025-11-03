"""
Crop-specific input requirements for smallholder farming per acre
Based on typical agricultural practices in Uganda
"""

# Input requirements per acre for different crops
# Units: 
#   - Seed: kg for traditional crops, or units (suckers/cuttings/seedlings) for others
#   - Fertilizer: kg per acre
#   - Herbicide: litres per acre
#   - Pesticide: litres per acre
#   - Labor: days per acre
CROP_REQUIREMENTS = {
    'Maize': {
        'seed': 20,  # kg per acre
        'fertilizer': 100,  # kg per acre (NPK or Urea)
        'herbicide': 2,  # litres per acre
        'pesticide': 1.5,  # litres per acre
        'labor': 12  # days per acre (planting, weeding, harvesting)
    },
    'Beans': {
        'seed': 25,  # kg per acre
        'fertilizer': 50,  # kg per acre (less fertilizer needed)
        'herbicide': 1.5,  # litres per acre
        'pesticide': 1,  # litres per acre
        'labor': 15  # days per acre (more labor for harvesting)
    },
    'Coffee': {
        'seed': 1200,  # Coffee uses seedlings (approximately 1200 seedlings per acre for smallholder)
        'fertilizer': 150,  # kg per acre (coffee needs more fertilizer)
        'herbicide': 3,  # litres per acre
        'pesticide': 2.5,  # litres per acre (coffee is more prone to pests)
        'labor': 25  # days per acre (labor-intensive crop)
    },
    'Rice': {
        'seed': 30,  # kg per acre (more seed needed)
        'fertilizer': 120,  # kg per acre
        'herbicide': 2.5,  # litres per acre
        'pesticide': 2,  # litres per acre
        'labor': 20  # days per acre (labor-intensive)
    },
           'Cassava': {
               'seed': 400,  # Cassava uses whole stems (approximately 400 stems per acre, each stem produces multiple cuttings)
        'fertilizer': 80,  # kg per acre
        'herbicide': 2,  # litres per acre
        'pesticide': 1.5,  # litres per acre
        'labor': 18  # days per acre
    },
    'Matooke': {
        'seed': 400,  # Matooke uses suckers (approximately 400 suckers per acre)
        'fertilizer': 90,  # kg per acre
        'herbicide': 2,  # litres per acre
        'pesticide': 1.5,  # litres per acre
        'labor': 22  # days per acre (more labor for harvesting)
    },
    # Alternative names
    'Plantain': {
        'seed': 400,  # Plantain uses suckers
        'fertilizer': 90,
        'herbicide': 2,
        'pesticide': 1.5,
        'labor': 22
    }
}

def get_crop_requirements(crop_name):
    """
    Get input requirements for a specific crop
    
    Args:
        crop_name: Name of the crop
        
    Returns:
        dict: Input requirements per acre
    """
    # Try exact match first
    if crop_name in CROP_REQUIREMENTS:
        return CROP_REQUIREMENTS[crop_name]
    
    # Try case-insensitive match
    crop_lower = crop_name.lower()
    for crop, requirements in CROP_REQUIREMENTS.items():
        if crop.lower() == crop_lower:
            return requirements
    
    # Default to Maize if crop not found
    return CROP_REQUIREMENTS['Maize']

def calculate_total_costs_per_acre(unit_prices, crop_name):
    """
    Calculate total costs per acre based on unit prices and crop requirements
    
    Args:
        unit_prices: dict with keys: seed_price_per_kg, fertilizer_price_per_kg,
                    herbicide_price_per_litre, pesticide_price_per_litre, labor_cost_per_day
        crop_name: Name of the crop
        
    Returns:
        dict: Total costs per input type and total per acre
    """
    requirements = get_crop_requirements(crop_name)
    
    # For crops that don't use traditional seeds, we need to adjust the calculation
    # Matooke, Cassava, Coffee use suckers/cuttings/seedlings which have different pricing
    # Use a multiplier or separate price for these materials
    crop_type = crop_name.lower()
    
    if crop_type in ['matooke', 'plantain']:
        # Matooke uses suckers - cost per sucker is typically 200-500 UGX
        # Use estimated cost per sucker based on typical smallholder prices
        sucker_cost_per_unit = unit_prices['seed_price_per_kg'] * 0.1  # Approximate sucker cost
        seed_cost = sucker_cost_per_unit * requirements['seed']
    elif crop_type == 'cassava':
        # Cassava uses whole stems - cost per stem is typically 500-1000 UGX (each stem produces multiple cuttings)
        stem_cost_per_unit = unit_prices['seed_price_per_kg'] * 0.2  # Approximate stem cost
        seed_cost = stem_cost_per_unit * requirements['seed']
    elif crop_type == 'coffee':
        # Coffee uses seedlings - cost per seedling is typically 500-1000 UGX
        seedling_cost_per_unit = unit_prices['seed_price_per_kg'] * 0.15  # Approximate seedling cost
        seed_cost = seedling_cost_per_unit * requirements['seed']
    else:
        # Traditional seed crops (Maize, Beans, Rice)
        seed_cost = unit_prices['seed_price_per_kg'] * requirements['seed']
    
    fertilizer_cost = unit_prices['fertilizer_price_per_kg'] * requirements['fertilizer']
    herbicide_cost = unit_prices['herbicide_price_per_litre'] * requirements['herbicide']
    pesticide_cost = unit_prices['pesticide_price_per_litre'] * requirements['pesticide']
    labor_cost = unit_prices['labor_cost_per_day'] * requirements['labor']
    
    total_cost_per_acre = (
        seed_cost +
        fertilizer_cost +
        herbicide_cost +
        pesticide_cost +
        labor_cost
    )
    
    # Determine seed unit label and quantity
    if crop_type in ['matooke', 'plantain']:
        seed_unit_label = 'suckers'
        seed_quantity_display = requirements['seed']
    elif crop_type == 'cassava':
        seed_unit_label = 'stems'
        seed_quantity_display = requirements['seed']
    elif crop_type == 'coffee':
        seed_unit_label = 'seedlings'
        seed_quantity_display = requirements['seed']
    else:
        seed_unit_label = 'kg'
        seed_quantity_display = requirements['seed']
    
    return {
        'seed_cost': seed_cost,
        'fertilizer_cost': fertilizer_cost,
        'herbicide_cost': herbicide_cost,
        'pesticide_cost': pesticide_cost,
        'labor_cost': labor_cost,
        'total_cost_per_acre': total_cost_per_acre,
        'requirements': requirements,
        'unit_prices': unit_prices,
        'seed_unit_label': seed_unit_label,
        'seed_quantity_display': seed_quantity_display
    }

