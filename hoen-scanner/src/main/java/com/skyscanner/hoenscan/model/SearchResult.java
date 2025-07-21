package com.skyscanner.hoenscan.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class SearchResult {
    @JsonProperty("city")
    private String city;

    @JsonProperty("kind")
    private String kind;

    @JsonProperty("title")
    private String title;

    // Default constructor for Jackson
    public SearchResult() {}

    public SearchResult(String city, String kind, String title) {
        this.city = city;
        this.kind = kind;
        this.title = title;
    }

    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    public String getKind() {
        return kind;
    }

    public void setKind(String kind) {
        this.kind = kind;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    @Override
    public String toString() {
        return "SearchResult{" +
                "city='" + city + '\'' +
                ", kind='" + kind + '\'' +
                ", title='" + title + '\'' +
                '}';
    }
}
