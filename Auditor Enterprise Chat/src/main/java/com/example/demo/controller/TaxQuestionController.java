package com.example.demo.controller;

import com.example.demo.service.OpenAIService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@CrossOrigin(origins = "*", allowedHeaders = "*")
@RestController
@RequestMapping("/api")
public class TaxQuestionController {

    @Autowired
    private OpenAIService openAIService;

    @PostMapping("/ask")
    public String askGPT3(@RequestBody Map<String, Object> data) {
        String promptText = (String) data.get("prompt");
        // Here, the actual interaction with the OpenAI service happens
        return openAIService.askGPT3(promptText);
    }
}
