package com.example.demo.service;

import org.apache.http.HttpHeaders;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.stereotype.Service;
import org.json.JSONObject;

@Service
public class OpenAIService {

    private static final String API_URL = "https://api.openai.com/v1/engines/davinci/completions";

    public String askGPT3(String prompt) {
        // Use your actual OpenAI API key here
        String apiKey = "sk-xxxxxxxxxxxxxxxxxxxxxxxx";

        try (CloseableHttpClient client = HttpClients.createDefault()) {
            HttpPost httpPost = new HttpPost(API_URL);

            // Set headers for authorization and content type
            httpPost.setHeader(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey);
            httpPost.setHeader(HttpHeaders.CONTENT_TYPE, "application/json");

            // Create JSON object with the text prompt
            JSONObject json = new JSONObject();
            json.put("prompt", prompt);
            json.put("max_tokens", 550); // Limit the response length// Adjust based on how lengthy you expect the response to be
            json.put("temperature", 0.2);

            // Set the request entity
            httpPost.setEntity(new StringEntity(json.toString()));

            // Execute the request
            try (CloseableHttpResponse response = client.execute(httpPost)) {
                // Parse and return the response
                String responseString = EntityUtils.toString(response.getEntity());
                JSONObject responseObject = new JSONObject(responseString);

                if (responseObject.has("choices")) {
                    String textResponse = responseObject.getJSONArray("choices").getJSONObject(0).getString("text").trim();
                    return textResponse;
                }
            }
        } catch (Exception e) {
            // Handle the exception appropriately
            e.printStackTrace();
        }
        return "Error occurred.";
    }
}
