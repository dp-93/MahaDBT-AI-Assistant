from fastmcp import FastMCP

# 1. Create the server
mcp = FastMCP(name="Maha Schemes Evaluator")

# 2. Add a tool using the decorator
@mcp.tool
def calculate_hostel_allowance(region: str) -> str:
    """Calculates the 10-month hostel allowance for a student based on their region."""
    urban_regions = ["Mumbai", "MMRDA", "Pune", "PMRDA", "Nagpur", "Aurangabad"]
    
    if any(urban.lower() in region.lower() for urban in urban_regions):
        return f"Student in {region} is eligible for Rs. 30,000 for 10 months."
    else:
        return f"Student in {region} (Other Region) is eligible for Rs. 20,000 for 10 months."

if __name__ == "__main__":
    print("🚀 Starting MCP Server...")
    mcp.run()