import pytest
from integration.conftest import run_cli


@pytest.fixture(params=[("gpt-4o-mini", "openai"), ("llama3.2", "ollama")])
def model_config(request):
    """Fixture to test with different models."""
    return request.param


def test_llm_last_basic_functionality(integration_setup, model_config):
    """Test basic llm_last functionality without GROUP BY."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    test_model_name = f"test-last-model_{model_name}"
    create_model_query = (
        f"CREATE MODEL('{test_model_name}', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE products (
        id INTEGER,
        name VARCHAR,
        price DECIMAL,
        rating DECIMAL
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    insert_data_query = """
                        INSERT INTO products
                        VALUES (1, 'Budget Phone', 199.99, 3.5),
                               (2, 'Premium Phone', 999.99, 4.8),
                               (3, 'Mid-range Phone', 499.99, 4.2); \
                        """
    run_cli(duckdb_cli_path, db_path, insert_data_query)

    query = (
            """
                SELECT llm_last(
                           {'model_name': '"""
            + test_model_name
            + """'},
            {'prompt': 'Which product offers the worst value for money? Return the ID number only.', 'context_columns': [{'data': name}, {'data': price::VARCHAR}, {'data': rating::VARCHAR}]}
        ) AS worst_value_product
            FROM products; \
            """
    )
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode == 0, f"Query failed with error: {result.stderr}"
    assert "worst_value_product" in result.stdout.lower()
    lines = result.stdout.strip().split("\n")
    assert len(lines) >= 2, "Expected at least header and one result row"


def test_llm_last_with_group_by(integration_setup, model_config):
    """Test llm_last with GROUP BY clause."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    test_model_name = f"test-last-group_{model_name}"
    create_model_query = (
        f"CREATE MODEL('{test_model_name}', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE restaurant_reviews (
        id INTEGER,
        city VARCHAR,
        restaurant_name VARCHAR,
        rating INTEGER,
        review VARCHAR
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    insert_data_query = """
                        INSERT INTO restaurant_reviews
                        VALUES (1, 'New York', 'Pizza Palace', 5, 'Amazing pizza with fresh ingredients'),
                               (2, 'New York', 'Burger Barn', 2, 'Dry burger and cold fries'),
                               (3, 'Chicago', 'Deep Dish Delight', 4, 'Good deep dish but service was slow'),
                               (4, 'Chicago', 'Taco Trouble', 1, 'Terrible food and poor hygiene'); \
                        """
    run_cli(duckdb_cli_path, db_path, insert_data_query)

    query = (
            """
            SELECT *
            FROM duckdb_secrets();
            SELECT city,
                   llm_last(
                       {'model_name': '"""
            + test_model_name
            + """'},
            {'prompt': 'Which restaurant has the worst reviews in this city? Return the ID number only.', 'context_columns': [{'data': restaurant_name}, {'data': rating::VARCHAR}, {'data': review}]}
        ) AS worst_restaurant_id
            FROM restaurant_reviews
            GROUP BY city
            ORDER BY city; \
            """
    )
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode == 0, f"Query failed with error: {result.stderr}"
    lines = result.stdout.strip().split("\n")
    # Should have header + 2 city groups (Chicago, New York)
    assert len(lines) >= 3, f"Expected at least 3 lines, got {len(lines)}"
    assert "chicago" in result.stdout.lower() or "new york" in result.stdout.lower()


def test_llm_last_with_batch_processing(integration_setup, model_config):
    """Test llm_last with batch size configuration."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    test_model_name = f"test-last-batch_{model_name}"
    create_model_query = (
        f"CREATE MODEL('{test_model_name}', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE service_providers (
        id INTEGER,
        company_name VARCHAR,
        service_type VARCHAR,
        customer_rating DECIMAL,
        response_time_hours INTEGER,
        price_per_hour DECIMAL,
        reliability_score INTEGER
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    insert_data_query = """
                        INSERT INTO service_providers
                        VALUES (1, 'QuickFix Pro', 'Plumbing', 4.2, 2, 85.00, 88),
                               (2, 'Slow Solutions', 'Plumbing', 2.8, 8, 65.00, 45),
                               (3, 'Expensive Experts', 'Plumbing', 3.5, 4, 150.00, 70),
                               (4, 'Unreliable Repairs', 'Plumbing', 2.1, 12, 75.00, 30),
                               (5, 'Budget Plumbers', 'Plumbing', 3.8, 6, 55.00, 65); \
                        """
    run_cli(duckdb_cli_path, db_path, insert_data_query)

    query = (
            """
                SELECT llm_last(
                           {'model_name': '"""
            + test_model_name
            + """', 'batch_size': 3},
            {'prompt': 'Which service provider offers the worst overall service considering rating, response time, and reliability? Return the ID number only.', 'context_columns': [{'data': company_name}, {'data': customer_rating::VARCHAR}, {'data': response_time_hours::VARCHAR}, {'data': price_per_hour::VARCHAR}, {'data': reliability_score::VARCHAR}]}
        ) AS worst_service_provider
            FROM service_providers; \
            """
    )
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode == 0, f"Query failed with error: {result.stderr}"
    assert "worst_service_provider" in result.stdout.lower()


def test_llm_last_with_model_parameters(integration_setup, model_config):
    """Test llm_last with custom model parameters."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    test_model_name = f"test-last-params_{model_name}"
    create_model_query = (
        f"CREATE MODEL('{test_model_name}', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE movie_reviews (
        id INTEGER,
        title VARCHAR,
        genre VARCHAR,
        rating DECIMAL,
        review VARCHAR,
        box_office INTEGER
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    insert_data_query = """
                        INSERT INTO movie_reviews
                        VALUES (1, 'Action Hero', 'Action', 8.5, 'Thrilling action sequences with great stunts',
                                150000000),
                               (2, 'Romance Story', 'Romance', 6.2, 'Predictable plot but decent acting', 45000000),
                               (3, 'Horror Night', 'Horror', 4.1, 'Poor script and unconvincing special effects',
                                25000000),
                               (4, 'Comedy Disaster', 'Comedy', 3.2, 'Unfunny jokes and terrible timing', 15000000); \
                        """
    run_cli(duckdb_cli_path, db_path, insert_data_query)

    query = (
            """
                SELECT llm_last(
                           {'model_name': '"""
            + test_model_name
            + """', 'tuple_format': 'Markdown',
                                                          'model_parameters': '{"temperature": 0.1}'},
            {'prompt': 'Which movie was the biggest disappointment considering its budget and reviews? Return the ID number only.', 'context_columns': [{'data': title}, {'data': genre}, {'data': rating::VARCHAR}, {'data': review}, {'data': box_office::VARCHAR}]}
        ) AS biggest_disappointment
            FROM movie_reviews; \
            """
    )
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode == 0, f"Query failed with error: {result.stderr}"
    assert "biggest_disappointment" in result.stdout.lower()


def test_llm_last_multiple_criteria(integration_setup, model_config):
    """Test llm_last with multiple evaluation criteria."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    test_model_name = f"test-last-multi_{model_name}"
    create_model_query = (
        f"CREATE MODEL('{test_model_name}', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE housing_options (
        id INTEGER,
        address VARCHAR,
        price_per_month INTEGER,
        size_sqft INTEGER,
        commute_time_minutes INTEGER,
        neighborhood_rating INTEGER,
        condition_score INTEGER
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    insert_data_query = """
                        INSERT INTO housing_options
                        VALUES (1, '123 Downtown Ave', 2500, 800, 15, 85, 90),
                               (2, '456 Suburb St', 1800, 1200, 45, 92, 88),
                               (3, '789 Highway Rd', 1200, 600, 60, 65, 70),
                               (4, '321 Noisy Blvd', 2200, 750, 20, 40, 60); \
                        """
    run_cli(duckdb_cli_path, db_path, insert_data_query)

    query = (
            """
                SELECT llm_last(
                           {'model_name': '"""
            + test_model_name
            + """'},
                    {'prompt': 'Which housing option offers the worst value considering price, location quality, commute time, and living conditions? Return the ID number only.',
                    'context_columns': [{'data': address}, {'data': price_per_month::VARCHAR}, {'data': size_sqft::VARCHAR}, {'data': commute_time_minutes::VARCHAR}, {'data': neighborhood_rating::VARCHAR}, {'data': condition_score::VARCHAR}]}
        ) AS worst_housing_value
            FROM housing_options; \
            """
    )
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode == 0, f"Query failed with error: {result.stderr}"
    assert "worst_housing_value" in result.stdout.lower()


def test_llm_last_empty_table(integration_setup, model_config):
    """Test llm_last behavior with empty table."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    test_model_name = f"test-last-empty_{model_name}"
    create_model_query = (
        f"CREATE MODEL('{test_model_name}', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE empty_products (
        id INTEGER,
        name VARCHAR
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    query = (
            """
                SELECT llm_last(
                           {'model_name': '"""
            + test_model_name
            + """'},
                    {'prompt': 'Select the worst product', 'context_columns': [{'data': name}]}
        ) AS selected
            FROM empty_products; \
            """
    )
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode == 0, f"Query failed with error: {result.stderr}"
    # Should return empty result or no rows
    lines = result.stdout.strip().split("\n")
    assert len(lines) <= 2, "Expected at most header line for empty table"


def test_llm_last_error_handling_invalid_model(integration_setup):
    """Test error handling with non-existent model."""
    duckdb_cli_path, db_path = integration_setup

    create_table_query = """
    CREATE OR REPLACE TABLE test_data (
        id INTEGER,
        text VARCHAR
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    insert_data_query = """
                        INSERT INTO test_data
                        VALUES (1, 'Test content'); \
                        """
    run_cli(duckdb_cli_path, db_path, insert_data_query)

    query = """
            SELECT llm_last(
                       {'model_name': 'non-existent-model'},
        {'prompt': 'Select the worst item', 'context_columns': [{'data': text}]}
    ) AS result
            FROM test_data; \
            """
    result = run_cli(duckdb_cli_path, db_path, query)

    assert (
            result.returncode != 0
            or "error" in result.stderr.lower()
            or "Error" in result.stdout
    )


def test_llm_last_error_handling_empty_prompt(integration_setup, model_config):
    """Test error handling with empty prompt."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    test_model_name = f"test-last-empty-prompt_{model_name}"
    create_model_query = (
        f"CREATE MODEL('{test_model_name}', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE test_data (
        id INTEGER,
        text VARCHAR
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    insert_data_query = """
                        INSERT INTO test_data
                        VALUES (1, 'Test content'); \
                        """
    run_cli(duckdb_cli_path, db_path, insert_data_query)

    query = (
            """
                SELECT llm_last(
                           {'model_name': '"""
            + test_model_name
            + """'},
        {'prompt': '', 'context_columns': [{'data': text}]}
    ) AS result
            FROM test_data; \
            """
    )
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode != 0


def test_llm_last_error_handling_missing_arguments(integration_setup, model_config):
    """Test error handling with missing required arguments."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    test_model_name = f"test-last-missing-args_{model_name}"
    create_model_query = (
        f"CREATE MODEL('{test_model_name}', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    # Test with only 1 argument (should fail since llm_last requires 2)
    query = (
            """
        SELECT llm_last(
            {'model_name': '"""
            + test_model_name
            + """'}
    ) AS result;
    """
    )
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode != 0, "Expected error for missing second argument"


def test_llm_last_with_special_characters(integration_setup, model_config):
    """Test llm_last with special characters and unicode."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    test_model_name = f"test-last-unicode_{model_name}"
    create_model_query = (
        f"CREATE MODEL('{test_model_name}', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE travel_destinations (
        id INTEGER,
        destination VARCHAR,
        cost_per_day INTEGER,
        safety_rating INTEGER,
        weather_score INTEGER,
        description VARCHAR
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    insert_data_query = """
                        INSERT INTO travel_destinations
                        VALUES (1, 'Paris, France 🇫🇷', 150, 85, 70, 'Beautiful city with café culture and art'),
                               (2, 'Dangerous Zone ⚠️', 250, 25, 60, 'High crime area with poor infrastructure'),
                               (3, '東京, Japan 🇯🇵', 200, 95, 80, 'Modern metropolis with excellent transportation'); \
                        """
    run_cli(duckdb_cli_path, db_path, insert_data_query)

    query = (
            """
                SELECT llm_last(
                           {'model_name': '"""
            + test_model_name
            + """'},
                    {'prompt': 'Which destination is the least safe and appealing for tourists? Return the ID number only.', 'context_columns': [{'data': destination}, {'data': cost_per_day::VARCHAR}, {'data': safety_rating::VARCHAR}, {'data': weather_score::VARCHAR}, {'data': description}]}
        ) AS least_appealing_destination
            FROM travel_destinations; \
            """
    )
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode == 0, f"Query failed with error: {result.stderr}"
    assert "least_appealing_destination" in result.stdout.lower()


def _test_llm_last_performance_large_dataset(integration_setup, model_config):
    """Performance test with larger dataset - commented out with underscore prefix for optional execution."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    test_model_name = f"test-last-perf_{model_name}"
    create_model_query = (
        f"CREATE MODEL('{test_model_name}', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE large_product_pool AS
    SELECT
        i as id,
        'Product ' || i as name,
        'Category ' || (i % 4) as category,
        (10 - (i % 10)) as quality_score
    FROM range(1, 41) t(i);
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    query = (
            """
                SELECT category,
                       llm_last(
                           {'model_name': '"""
            + test_model_name
            + """', 'batch_size': 8},
                     {'prompt': 'Which product has the lowest quality in this category? Return the ID number only.', 'context_columns': [{'data': name}, {'data': quality_score::VARCHAR}]}
        ) AS worst_product
            FROM large_product_pool
            GROUP BY category
            ORDER BY category LIMIT 3; \
            """
    )
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode == 0, f"Query failed with error: {result.stderr}"
    lines = result.stdout.strip().split("\n")
    assert len(lines) >= 4, (
        f"Expected at least 4 lines (header + 3 categories), got {len(lines)}"
    )
    assert "category" in result.stdout.lower()


def test_llm_last_with_image_integration(integration_setup, model_config):
    """Test llm_last with image data integration."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    create_model_query = (
        f"CREATE MODEL('test-image-last-model', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE flower_images (
        id INTEGER,
        flower_name VARCHAR,
        image_url VARCHAR,
        color VARCHAR,
        blooming_season VARCHAR
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    # Insert data with Unsplash flower image URLs
    insert_data_query = """
                        INSERT INTO flower_images
                        VALUES (1, 'Rose', 'https://images.unsplash.com/photo-1496062031456-07b8f162a322?w=400',
                                'Red', 'Spring'),
                               (2, 'Tulip', 'https://images.unsplash.com/photo-1520763185298-1b434c919102?w=400',
                                'Yellow', 'Spring'),
                               (3, 'Sunflower',
                                'https://images.unsplash.com/photo-1597848212624-a19eb35e2651?w=400',
                                'Yellow', 'Summer');
                        """
    run_cli(duckdb_cli_path, db_path, insert_data_query)

    query = """
            SELECT llm_last(
                       {'model_name': 'test-image-last-model'},
            {
                'prompt': 'Which flower image shows the latest blooming season? Return the flower name only.',
                'context_columns': [
                    {'data': flower_name},
                    {'data': image_url, 'type': 'image'},
                    {'data': blooming_season}
                ]
            }
        ) AS latest_blooming_flower
            FROM flower_images;
            """
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode == 0, f"Query failed with error: {result.stderr}"
    assert "latest_blooming_flower" in result.stdout.lower()
    assert len(result.stdout.strip().split("\n")) >= 2


def test_llm_last_image_with_group_by(integration_setup, model_config):
    """Test llm_last with images and GROUP BY clause."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    create_model_query = (
        f"CREATE MODEL('test-image-group-last', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE car_images (
        id INTEGER,
        car_model VARCHAR,
        image_url VARCHAR,
        brand VARCHAR,
        year INTEGER,
        price_category VARCHAR
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    # Insert data with Unsplash car image URLs
    insert_data_query = """
                        INSERT INTO car_images
                        VALUES (1, 'Sedan A', 'https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=400',
                                'Brand X', 2020, 'Mid-range'),
                               (2, 'SUV B', 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400',
                                'Brand Y', 2019, 'High-end'),
                               (3, 'Hatchback C', 'https://images.unsplash.com/photo-1532298229144-0ec0c57515c7?w=400',
                                'Brand Z', 2021, 'Budget'),
                               (4, 'Sports Car D', 'https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=400',
                                'Brand X', 2022, 'Luxury');
                        """
    run_cli(duckdb_cli_path, db_path, insert_data_query)

    query = """
            SELECT brand,
                   llm_last(
                       {'model_name': 'test-image-group-last'},
            {
                'prompt': 'Which car from this brand is the oldest model? Return the car model only.',
                'context_columns': [
                    {'data': car_model},
                    {'data': image_url, 'type': 'image'},
                    {'data': year::VARCHAR}
                ]
            }
        ) AS oldest_car
            FROM car_images
            GROUP BY brand
            ORDER BY brand;
            """
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode == 0, f"Query failed with error: {result.stderr}"
    lines = result.stdout.strip().split("\n")
    assert len(lines) >= 4, (
        f"Expected at least 4 lines (header + 3 brands), got {len(lines)}"
    )
    assert "oldest_car" in result.stdout.lower()


def test_llm_last_image_batch_processing(integration_setup, model_config):
    """Test llm_last with multiple images in batch processing."""
    duckdb_cli_path, db_path = integration_setup
    model_name, provider = model_config

    create_model_query = (
        f"CREATE MODEL('test-image-batch-last', '{model_name}', '{provider}');"
    )
    run_cli(duckdb_cli_path, db_path, create_model_query)

    create_table_query = """
    CREATE OR REPLACE TABLE restaurant_images (
        id INTEGER,
        restaurant_name VARCHAR,
        image_url VARCHAR,
        cuisine_type VARCHAR,
        rating DECIMAL(3,1),
        price_level INTEGER
    );
    """
    run_cli(duckdb_cli_path, db_path, create_table_query)

    # Insert data with Unsplash restaurant image URLs
    insert_data_query = """
                        INSERT INTO restaurant_images
                        VALUES (1, 'Italian Bistro',
                                'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400', 'Italian', 4.2,
                                2),
                               (2, 'Sushi Bar', 'https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=400',
                                'Japanese', 4.5, 3),
                               (3, 'Burger Joint', 'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400',
                                'American', 3.8, 1),
                               (4, 'French Cafe', 'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400',
                                'French', 4.7, 4);
                        """
    run_cli(duckdb_cli_path, db_path, insert_data_query)

    query = """
            SELECT cuisine_type,
                   llm_last(
                       {'model_name': 'test-image-batch-last', 'batch_size': 2},
            {
                'prompt': 'Which restaurant in this cuisine type has the lowest rating? Return the restaurant name only.',
                'context_columns': [
                    {'data': restaurant_name},
                    {'data': image_url, 'type': 'image'},
                    {'data': rating::VARCHAR}
                ]
            }
        ) AS lowest_rated_restaurant
            FROM restaurant_images
            GROUP BY cuisine_type
            ORDER BY cuisine_type;
            """
    result = run_cli(duckdb_cli_path, db_path, query)

    assert result.returncode == 0, f"Query failed with error: {result.stderr}"
    lines = result.stdout.strip().split("\n")
    assert len(lines) >= 4, (
        f"Expected at least 4 lines (header + 3 cuisines), got {len(lines)}"
    )
    assert "lowest_rated_restaurant" in result.stdout.lower()
