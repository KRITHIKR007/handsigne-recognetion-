package com.skyscanner.hoenscan.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Search {
    @JsonProperty("city")
    private String city;

    // Default constructor for Jackson
    public Search() {}

    public Search(String city) {
        this.city = city;
    }

    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    @Override
    public String toString() {
        return "Search{" +
                "city='" + city + '\'' +
                '}';
    }
}
