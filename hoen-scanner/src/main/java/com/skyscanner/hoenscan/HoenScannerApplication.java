package com.skyscanner.hoenscan;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.skyscanner.hoenscan.model.SearchResult;
import com.skyscanner.hoenscan.resources.SearchResource;
import io.dropwizard.Application;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class HoenScannerApplication extends Application<HoenScannerConfiguration> {

    public static void main(String[] args) throws Exception {
        System.out.println("Welcome to Hoen Scanner!");
        new HoenScannerApplication().run(args);
    }

    @Override
    public String getName() {
        return "hoen-scanner";
    }

    @Override
    public void initialize(Bootstrap<HoenScannerConfiguration> bootstrap) {
        // Add any additional configuration here
    }

    @Override
    public void run(HoenScannerConfiguration configuration, Environment environment) throws Exception {
        // Load JSON data from resources
        List<SearchResult> allResults = loadSearchResults();
        
        // Register the search resource
        final SearchResource searchResource = new SearchResource(allResults);
        environment.jersey().register(searchResource);
        
        System.out.println("Hoen Scanner service is running on http://localhost:8080");
        System.out.println("POST endpoint available at: http://localhost:8080/search");
        System.out.println("Loaded " + allResults.size() + " search results from JSON files.");
    }

    private List<SearchResult> loadSearchResults() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        List<SearchResult> allResults = new ArrayList<>();

        // Load rental cars
        try (InputStream rentalCarsStream = getClass().getResourceAsStream("/rental_cars.json")) {
            if (rentalCarsStream != null) {
                List<SearchResult> rentalCars = mapper.readValue(rentalCarsStream, 
                        new TypeReference<List<SearchResult>>() {});
                allResults.addAll(rentalCars);
                System.out.println("Loaded " + rentalCars.size() + " rental cars");
            } else {
                System.err.println("Warning: rental_cars.json not found in resources");
            }
        }

        // Load hotels
        try (InputStream hotelsStream = getClass().getResourceAsStream("/hotels.json")) {
            if (hotelsStream != null) {
                List<SearchResult> hotels = mapper.readValue(hotelsStream, 
                        new TypeReference<List<SearchResult>>() {});
                allResults.addAll(hotels);
                System.out.println("Loaded " + hotels.size() + " hotels");
            } else {
                System.err.println("Warning: hotels.json not found in resources");
            }
        }

        return allResults;
    }
}
