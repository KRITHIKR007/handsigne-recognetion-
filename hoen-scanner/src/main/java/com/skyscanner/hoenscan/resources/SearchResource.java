package com.skyscanner.hoenscan.resources;

import java.util.List;
import java.util.stream.Collectors;

import javax.ws.rs.BadRequestException;
import javax.ws.rs.Consumes;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

import com.skyscanner.hoenscan.model.Search;
import com.skyscanner.hoenscan.model.SearchResult;

@Path("/search")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class SearchResource {

    private final List<SearchResult> searchResults;

    public SearchResource(List<SearchResult> searchResults) {
        this.searchResults = searchResults;
    }

    @POST
    public List<SearchResult> search(Search search) {
        if (search == null || search.getCity() == null) {
            throw new BadRequestException("City is required");
        }

        String searchCity = search.getCity().toLowerCase().trim();
        
        return searchResults.stream()
                .filter(result -> result.getCity() != null && 
                        result.getCity().toLowerCase().equals(searchCity))
                .collect(Collectors.toList());
    }
}
