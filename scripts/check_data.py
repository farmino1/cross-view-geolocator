import os

# Check exact structure
for city in ["seattle", "london", "tokyo", "sydney"]:
    print(f"\n{city}/")
    for item in os.listdir(f"data/{city}"):
        full = f"data/{city}/{item}"
        if os.path.isdir(full):
            count = len(os.listdir(full))
            print(f"  {item}/ ({count} files)")
            # Show first few files
            for f in sorted(os.listdir(full))[:3]:
                print(f"    {f}")
        else:
            size = os.path.getsize(full)
            print(f"  {item} ({size} bytes)")
