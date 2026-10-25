import asyncio
import os
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

load_dotenv("../.env")
load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

print("URI configured:", bool(uri))
print("User configured:", bool(user))
print("Password configured:", bool(password))

async def test():
    if not all([uri, user, password]):
        print("Missing Neo4j configuration. Check variable names in your .env (do not share values).")
        return

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        await driver.verify_connectivity()
        print("Neo4j connectivity: OK")
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS ok")
            print("Query result:", await result.single())
    except Exception as exc:
        print(type(exc).__name__ + ":", str(exc))
    finally:
        await driver.close()

asyncio.run(test())
