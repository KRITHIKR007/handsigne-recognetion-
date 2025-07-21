# Hoen Scanner Microservice

A Dropwizard-based Java microservice for Skyscanner's Hoen Archipelago search system.

## Features

- Search for hotels and rental cars by city name
- RESTful API with JSON input/output
- Combines data from multiple JSON sources
- Built with Dropwizard framework and Jackson for JSON processing

## Prerequisites

- Java 17 or higher
- Maven 3.6 or higher

## Building the Application

```bash
mvn clean package
```

This will create a fat JAR file in the `target` directory.

## Running the Application

```bash
java -jar target/hoen-scanner-1.0.0.jar server config.yml
```

The application will start on `http://localhost:8080`.

## API Usage

### Search Endpoint

**URL:** `POST /search`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "city": "petalborough"
}
```

**Response:**
```json
[
  {
    "city": "petalborough",
    "kind": "hotel",
    "title": "Petalborough Grand Hotel"
  },
  {
    "city": "petalborough",
    "kind": "hotel",
    "title": "Lotus Inn Petalborough"
  },
  {
    "city": "petalborough",
    "kind": "rental_car",
    "title": "Petalborough Car Rentals - Economy"
  }
]
```

## Example Usage with curl

```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"city": "petalborough"}'
```

## Available Cities

The sample data includes the following cities:
- petalborough
- crystalpoint
- willowbrook
- suncrest
- moonhaven

## Project Structure

```
hoen-scanner/
├── src/
│   └── main/
│       ├── java/
│       │   └── com/
│       │       └── skyscanner/
│       │           └── hoenscan/
│       │               ├── model/
│       │               │   ├── Search.java
│       │               │   └── SearchResult.java
│       │               ├── resources/
│       │               │   └── SearchResource.java
│       │               ├── HoenScannerApplication.java
│       │               └── HoenScannerConfiguration.java
│       └── resources/
│           ├── hotels.json
│           └── rental_cars.json
├── config.yml
├── pom.xml
└── README.md
```
