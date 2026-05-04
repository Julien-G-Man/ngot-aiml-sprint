from pydantic import BaseModel, Field, ConfigDict 
from typing import Literal 
  
  
class RecommendRequest(BaseModel): 
    """Input: customer query and optional filters.""" 
    model_config = ConfigDict(extra='forbid') 
  
    query: str = Field( 
        ..., min_length=3, max_length=500, 
        description='Customer query or description of what they are looking for', 
        examples=['something cozy to wear in the rain', 'gift for a tech-savvy friend'], 
    ) 
    top_k: int = Field(default=5, ge=1, le=10, description='Number of recommendations') 
    category_filter: str | None = Field( 
        default=None, 
        description='Optional: filter to a specific category (Fashion, Electronics, etc.)', 
    ) 
    max_price_ghs: float | None = Field( 
        default=None, gt=0, 
        description='Optional: maximum price in Ghana Cedis', 
    ) 
  
  
class ProductRecommendation(BaseModel): 
    """A single recommended product.""" 
    product_id:       str 
    name:             str 
    description:      str 
    category:         str 
    price_ghs:        float 
    similarity_score: float   # Cosine similarity to the query (0-1) 
    rank:             int 
  
  
class RecommendResponse(BaseModel): 
    """Output: list of recommended products with metadata.""" 
    recommendations:   list[ProductRecommendation] 
    query:             str 
    total_returned:    int 
    model_used:        str 