import json
import time

from pydantic import BaseModel, Field

from lib.llm_parallelization.src.llm_parallelization.parallelization import (
    LLAMA_33_70B_INSTRUCT,
    HeterogeneousSchemaProcessor,
)

gpu_list = [0, 1, 2, 3]
multiplicity = 1
max_model_len = 10240
gpu_memory_utilization = 0.95
llm = LLAMA_33_70B_INSTRUCT


def example_1(processor):
    """
    Generate JSON schema from Pydantic model and use with processor.
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Usage with Pydantic-Generated Schema")
    print("=" * 70)

    class User(BaseModel):
        id: int = Field(description="Unique identifier for the user")
        name: str
        email: str = Field(pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        is_active: bool = True

    user_schema = User.model_json_schema()
    print("\nGenerated JSON Schema:")
    print(json.dumps(user_schema, indent=2))

    prompts = [
        "Create a user profile for John Doe with email john@example.com",
        "Generate user data for Jane Smith, ID 42, email jane@test.org",
    ]

    print("\nProcessing prompts with JSON schema...")

    start_time = time.time()
    results = processor.process_with_json_schema(prompts=prompts, json_schema=user_schema)
    parsed = processor.parse_results()
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Execution time: {elapsed_time:.2f} seconds")
    print(f"\nParsed {len(parsed)} results:")
    for i, result in enumerate(parsed):
        if isinstance(result, dict):
            print(f"  Result {i + 1}: {result}")
        else:
            print(f"  Result {i + 1}: Failed to parse")


def example_2(processor):
    """
    Create JSON schema manually without Pydantic.
    """
    print("\n" + "=" * 70)
    print("Example 2: Using Manually Created JSON Schema")
    print("=" * 70)

    location_schema = {
        "type": "object",
        "title": "Location",
        "properties": {
            "city": {"type": "string", "description": "Name of the city"},
            "country": {"type": "string", "description": "Name of the country"},
            "population": {"type": "integer", "description": "Population of the city"},
            "is_capital": {"type": "boolean", "description": "Whether the city is a capital"},
        },
        "required": ["city", "country"],
    }

    print("\nManual JSON Schema:")
    print(json.dumps(location_schema, indent=2))

    prompts = [
        "What is the capital of Canada?",
        "Tell me about Tokyo, Japan",
        "Provide information about Paris",
    ]

    print("\nProcessing prompts...")

    start_time = time.time()
    results = processor.process_with_json_schema(
        prompts=prompts, json_schema=location_schema, batch_size=3
    )
    parsed = processor.parse_results()
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Execution time: {elapsed_time:.2f} seconds")
    print(f"\nParsed {len(parsed)} location results:")
    for i, result in enumerate(parsed):
        if isinstance(result, dict):
            print(f"\n  Location {i + 1}:")
            print(f"    City: {result.get('city')}")
            print(f"    Country: {result.get('country')}")
            print(f"    Population: {result.get('population', 'N/A')}")
            print(f"    Is Capital: {result.get('is_capital', 'N/A')}")


def example_3(processor):
    """
    Switch between different JSON schemas in the same session.
    """
    print("\n" + "=" * 70)
    print("Example 3: Schema Switching - Multiple Schemas")
    print("=" * 70)

    person_schema = {
        "type": "object",
        "title": "Person",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "occupation": {"type": "string"},
        },
        "required": ["name", "age"],
    }

    product_schema = {
        "type": "object",
        "title": "Product",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "category": {"type": "string"},
            "in_stock": {"type": "boolean"},
        },
        "required": ["name", "price"],
    }

    start_time = time.time()

    print("\n1. Processing with Person schema...")
    person_prompts = ["Create a profile for a software engineer named Alice, age 30"]
    processor.process_with_json_schema(prompts=person_prompts, json_schema=person_schema)
    person_results = processor.parse_results()
    print(f"   Person result: {person_results[0]}")

    print("\n2. Processing with Product schema...")
    product_prompts = ["Create a product entry for a laptop priced at $999"]
    processor.process_with_json_schema(prompts=product_prompts, json_schema=product_schema)
    product_results = processor.parse_results()
    print(f"   Product result: {product_results[0]}")

    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds")


def example_4(processor):
    """
    Heterogeneous batch processing with different schemas.
    """
    print("\n" + "=" * 70)
    print("Example 4: Heterogeneous Batch Processing")
    print("=" * 70)

    person_schema = {
        "type": "object",
        "title": "Person",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "occupation": {"type": "string"},
        },
        "required": ["name", "age"],
    }

    company_schema = {
        "type": "object",
        "title": "Company",
        "properties": {
            "name": {"type": "string"},
            "industry": {"type": "string"},
            "employees": {"type": "integer"},
        },
        "required": ["name", "industry"],
    }

    location_schema = {
        "type": "object",
        "title": "Location",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "population": {"type": "integer"},
        },
        "required": ["city", "country"],
    }

    prompt_schema_pairs = [
        ("Create profile for John Doe, age 30, software engineer", person_schema),
        ("Info about Tesla in automotive industry with 100k employees", company_schema),
        ("Details about Tokyo, Japan with population 14 million", location_schema),
        ("Profile for Jane Smith, age 25, data scientist", person_schema),
        ("Info about Apple in technology with 150k employees", company_schema),
        ("Details about Paris, France with population 2 million", location_schema),
        ("Create profile for Bob Wilson, age 35, manager", person_schema),
        ("Info about Microsoft in software with 200k employees", company_schema),
    ]

    print(f"\nProcessing {len(prompt_schema_pairs)} prompts with heterogeneous batch...")

    start_time = time.time()
    results = processor.process_heterogeneous_batch(
        prompt_schema_pairs=prompt_schema_pairs, batch_size=10, preserve_order=True
    )
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Execution time: {elapsed_time:.2f} seconds")
    print("\n📋 Results (in original order):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result}")


def example_5(processor):
    """
    Use complex nested JSON schema with arrays and nested objects.
    """
    print("\n" + "=" * 70)
    print("Example 5: Advanced - Nested Schema with Arrays")
    print("=" * 70)

    company_schema = {
        "type": "object",
        "title": "Company",
        "properties": {
            "name": {"type": "string"},
            "founded_year": {"type": "integer"},
            "employees": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "position": {"type": "string"},
                        "salary": {"type": "number"},
                    },
                    "required": ["name", "position"],
                },
            },
            "headquarters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                "required": ["city"],
            },
        },
        "required": ["name", "founded_year"],
    }

    print("\nNested JSON Schema:")
    print(json.dumps(company_schema, indent=2))

    prompts = ["Create a tech company profile with 2-3 employees and headquarters information"]

    print("\nProcessing with nested schema...")

    start_time = time.time()
    processor.process_with_json_schema(prompts=prompts, json_schema=company_schema)
    results = processor.parse_results()
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Execution time: {elapsed_time:.2f} seconds")
    print("\nParsed company data:")
    print(json.dumps(results[0], indent=2))


def example_6(processor):
    """
    Parse results and validate against schema.
    """
    print("\n" + "=" * 70)
    print("Example 6: Validation Against Schema")
    print("=" * 70)

    book_schema = {
        "type": "object",
        "title": "Book",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "year": {"type": "integer"},
            "isbn": {"type": "string"},
        },
        "required": ["title", "author", "year"],
    }

    prompts = [
        "Create a book entry for '1984' by George Orwell, published 1949",
        "Add book information for 'The Hobbit' by J.R.R. Tolkien",
    ]

    start_time = time.time()
    processor.process_with_json_schema(prompts=prompts, json_schema=book_schema)
    parsed = processor.parse_results()
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Execution time: {elapsed_time:.2f} seconds")

    try:
        validation_results = processor.validate_against_schema(
            parsed_results=[r for r in parsed if isinstance(r, dict)], json_schema=book_schema
        )

        print("\nValidation Results:")
        for i, (result, valid) in enumerate(zip(parsed, validation_results)):
            status = "✓ Valid" if valid else "✗ Invalid"
            print(f"  Result {i + 1}: {status}")
            if valid:
                print(f"    {result}")
    except ImportError:
        print("\nNote: Install jsonschema for validation: pip install jsonschema")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING TIMED EXAMPLES")
    print("=" * 80)

    total_start_time = time.time()

    processor = HeterogeneousSchemaProcessor(
        gpu_list=gpu_list,
        multiplicity=multiplicity,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    example_1(processor)
    example_2(processor)
    example_3(processor)
    example_4(processor)
    example_5(processor)
    example_6(processor)

    processor.terminate()

    total_elapsed_time = time.time() - total_start_time

    print("\n" + "=" * 80)
    print(f"⏱️  TOTAL EXECUTION TIME: {total_elapsed_time:.2f} seconds")
    print("=" * 80 + "\n")
