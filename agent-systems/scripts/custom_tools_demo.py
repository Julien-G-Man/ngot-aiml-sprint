# Three patterns for creating LangChain tools

import os, json, math
from langchain.tools import tool
from langchain_core.tools import StructuredTool, BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# ── Pattern 1: @tool decorator (simplest) ────────────────────
# Great for simple functions. Docstring IS the description.
@tool
def calculate_orbital_period(semi_major_axis_km: float) -> float:
    """
    Calculate the orbital period of a satellite in minutes.
    Uses Kepler's third law: T = 2π × sqrt(a³/μ)
    where a = semi-major axis in km, μ = 398600.4418 km³/s²

    Use when: the user asks about orbital period, how long a satellite takes to orbit,
    or how many orbits per day a satellite completes.
    Input: semi_major_axis_km — distance from Earth's centre to the satellite in km.
    For LEO satellites: typically 6371 + 400 to 6371 + 2000 km
    """
    mu = 398600.4418  # Earth's gravitational parameter km³/s²
    T_seconds = 2 * math.pi * math.sqrt(semi_major_axis_km**3 / mu)
    return round(T_seconds / 60, 2)


# ── Pattern 2: StructuredTool with Pydantic input schema ─────
# Best for tools with multiple inputs — schemas are validated
class SatelliteQueryInput(BaseModel):
    norad_id:     int   = Field(..., description='NORAD catalogue number of the satellite')
    data_type:    str   = Field('all', description='Type: position, velocity, tle, or all')
    include_decay: bool = Field(False, description='Whether to include decay rate estimate')

def _get_satellite_data(norad_id: int, data_type: str = 'all', include_decay: bool = False) -> str:
    # Mock satellite database — in production, call space-track.org API
    MOCK_DB = {
        25544: {'name':'ISS','alt_km':420,'inclination':51.6,'period_min':92.7,'velocity_kms':7.7},
        48274: {'name':'Starlink2000','alt_km':550,'inclination':53,'period_min':95.6,'velocity_kms':7.6},
        43226: {'name':'Sentinel3B','alt_km':814,'inclination':98.6,'period_min':101.2,'velocity_kms':7.4},
    }
    sat = MOCK_DB.get(norad_id)
    if not sat:
        return json.dumps({'error': f'NORAD ID {norad_id} not found in database'})
    result = {'norad_id': norad_id, 'name': sat['name']}
    if data_type in ('position','all'):
        result['altitude_km']    = sat['alt_km']
        result['inclination_deg']= sat['inclination']
    if data_type in ('velocity','all'):
        result['velocity_km_s']  = sat['velocity_kms']
        result['period_min']     = sat['period_min']
    if include_decay:
        result['decay_rate_km_per_year'] = round(sat['alt_km'] * 0.002, 2)
    return json.dumps(result, indent=2)

satellite_tool = StructuredTool.from_function(
    func=_get_satellite_data,
    name='get_satellite_data',
    description=(
        'Retrieve orbital data for a specific satellite by NORAD ID. '
        'Returns altitude, inclination, velocity, and orbital period. '
        'Use for any question about a specific satellite. Known IDs: '
        'ISS=25544, Starlink-2000=48274, Sentinel-3B=43226.'
    ),
    args_schema=SatelliteQueryInput,
    return_direct=False,
)


print('Orbital period of ISS (alt ~420km):')
print(calculate_orbital_period.invoke({'semi_major_axis_km': 6371 + 420}))

print('\nISS data:')
print(satellite_tool.invoke({'norad_id': 25544, 'data_type': 'all', 'include_decay': True}))
