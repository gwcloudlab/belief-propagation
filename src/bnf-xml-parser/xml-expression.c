//
// Created by mjt5v on 3/6/17.
//
#include <libxml/parser.h>
#include <libxml/xpath.h>

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "xml-expression.h"

static xmlXPathObjectPtr get_node_set(xmlDocPtr doc, xmlChar * xpath){
    xmlXPathContextPtr context;
    xmlXPathObjectPtr result;

    context = xmlXPathNewContext(doc);
    if(context == NULL){
        printf("Error in xmlXPathNewContext\n");
        return NULL;
    }
    result = xmlXPathEvalExpression(xpath, context);
    if(result == NULL){
        printf("Error in xmlXPathEvalExpression\n");
        return NULL;
    }
    if(xmlXPathNodeSetIsEmpty(result->nodesetval)){
        xmlXPathFreeObject(result);
        printf("No Result\n");
        return NULL;
    }
    return result;
}

static xmlXPathObjectPtr get_subnode_set(xmlDocPtr doc, xmlChar * sub_xpath, xmlNodePtr node){
    xmlXPathContextPtr context;
    xmlXPathObjectPtr result;

    context = xmlXPathNewContext(doc);
    if(context == NULL){
        printf("Error in xmlXPathNewContext\n");
        return NULL;
    }
    result = xmlXPathNodeEval(node, sub_xpath, context);
    if(result == NULL){
        printf("Error in xmlXPathEvalExpression\n");
        return NULL;
    }
    if(xmlXPathNodeSetIsEmpty(result->nodesetval)){
        xmlXPathFreeObject(result);
        //printf("No Result\n");
        return NULL;
    }
    return result;
}

static void add_outcomes(xmlDocPtr doc, xmlNodePtr node, struct expression * expr){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    int i;
    xmlChar * value;

    struct expression * current;

    current = NULL;

    result = get_subnode_set(doc, (xmlChar *)".//OUTCOME/text()", node);
    assert(result);
    node_set = result->nodesetval;

    expr->int_value = node_set->nodeNr;

    for(i = 0; i < node_set->nodeNr; ++i){
        current = create_expression(VARIABLE_VALUES_LIST, current, NULL);
        value = xmlNodeListGetString(doc, node_set->nodeTab[i], 0);
        strncpy(current->value, (char *)value, CHAR_BUFFER_SIZE);
        xmlFree(value);
    }

    xmlXPathFreeObject(result);

    expr->left = current;
}

static void process_variable_declaration(xmlDocPtr doc, xmlNodePtr node, struct expression * expr){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    int i;
    xmlChar * name;

    result = get_subnode_set(doc, (xmlChar *)".//NAME/text()", node);
    assert(result);
    node_set = result->nodesetval;

    for(i = 0; i < node_set->nodeNr; ++i){
        name = xmlNodeListGetString(doc, node_set->nodeTab[i], 0);
        strncpy(expr->value, (char *)name, CHAR_BUFFER_SIZE);
        xmlFree(name);
        expr->left = create_expression(VARIABLE_CONTENT, create_expression(VARIABLE_OR_PROBABILITY, create_expression(VARIABLE_DISCRETE, NULL, NULL), NULL), NULL);
        add_outcomes(doc, node, expr->left->left->left);
    }
    xmlXPathFreeObject(result);
}

static void process_variables_list(xmlDocPtr doc, xmlNodePtr node, struct expression * expr){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    int i;
    xmlChar * name;
    struct expression * current;

    current = NULL;

    result = get_subnode_set(doc, (xmlChar *)".//FOR/text()", node);
    assert(result);
    node_set = result->nodesetval;
    for(i = 0; i < node_set->nodeNr; ++i){
        current = create_expression(PROBABILITY_VARIABLE_NAMES, current, NULL);
        name = xmlNodeListGetString(doc, node_set->nodeTab[i], 0);
        strncpy(current->value, (char *)name, CHAR_BUFFER_SIZE);
        xmlFree(name);
    }

    xmlXPathFreeObject(result);

    result = get_subnode_set(doc, (xmlChar *)".//GIVEN/text()", node);
    if(result != NULL) {
        node_set = result->nodesetval;

        for (i = 0; i < node_set->nodeNr; ++i) {
            current = create_expression(PROBABILITY_VARIABLE_NAMES, current, NULL);
            name = xmlNodeListGetString(doc, node_set->nodeTab[i], 0);
            strncpy(current->value, (char *) name, CHAR_BUFFER_SIZE);
            xmlFree(name);
        }
        xmlXPathFreeObject(result);
    }

    expr->left = current;
}

static void process_floats(xmlDocPtr doc, xmlNodePtr node, struct expression * expr){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    int i;
    xmlChar * value;
    char * split;
    struct expression * current;

    current = NULL;
    result = get_subnode_set(doc, (xmlChar *)".//TABLE/text()", node);
    assert(result);

    node_set = result->nodesetval;
    value = xmlNodeListGetString(doc, node_set->nodeTab[node_set->nodeNr - 1], 0);

    split = strtok((char *)value, " \t\n\r");
    while(split != NULL){
        if(strlen(split) > 0) {
            current = create_expression(FLOATING_POINT_LIST, current, NULL);
            sscanf(split, "%lf", &current->double_value);
        }
        split = strtok(NULL, " \t\n\r");
    }

    xmlXPathFreeObject(result);

    expr->left = current;
}

static void process_definition(xmlDocPtr doc, xmlNodePtr node, struct expression * expr){
    expr->left = create_expression(PROBABILITY_VARIABLES_LIST, NULL, NULL);
    process_variables_list(doc, node, expr->left);

    expr->right = create_expression(PROBABILITY_CONTENT, create_expression(PROBABILITY_CONTENT_LIST, create_expression(PROBABILITY_TABLE, NULL, NULL), NULL), NULL);
    process_floats(doc, node, expr->right->left->left);
}

static struct expression * process_definitions(xmlDocPtr doc){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    int i;

    struct expression * current;

    result = get_node_set(doc, (xmlChar *)"//NETWORK/DEFINITION");
    assert(result);

    current = NULL;

    node_set = result->nodesetval;
    for(i = 0; i < node_set->nodeNr; ++i){
        current = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, current, create_expression(PROBABILITY_DECLARATION, NULL, NULL));
        process_definition(doc, node_set->nodeTab[i], current->right);
    }

    xmlXPathFreeObject(result);

    return current;
}

static struct expression * process_variables(xmlDocPtr doc){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    int i;

    struct expression * current;

    result = get_node_set(doc, (xmlChar *)"//NETWORK/VARIABLE");
    assert(result);

    current = NULL;

    node_set = result->nodesetval;
    for(i = 0; i < node_set->nodeNr; ++i){
        current = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, current, create_expression(VARIABLE_DECLARATION, NULL, NULL));
        process_variable_declaration(doc, node_set->nodeTab[i], current->right);
    }

    xmlXPathFreeObject(result);

    return current;
}

struct expression * process_variables_and_probabilities(xmlDocPtr doc){
    struct expression * expr;

    expr = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, process_variables(doc), process_definitions(doc));

    return expr;
}

static struct expression * process_network(xmlDocPtr doc) {
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    int i;
    struct expression * expr;
    xmlChar * value;

    result = get_node_set(doc, (xmlChar *)"//NETWORK/NAME/text()");
    assert(result);

    expr = create_expression(NETWORK_DECLARATION, NULL, NULL);
    node_set = result->nodesetval;
    for(i = 0; i < node_set->nodeNr; ++i){
        value = xmlNodeListGetString(doc, node_set->nodeTab[i], 0);
        strncpy(expr->value, (char *)value, CHAR_BUFFER_SIZE);
        xmlFree(value);
    }
    xmlXPathFreeObject(result);

    return expr;
}

struct expression * parse_xml_file(const char * file_name){
    xmlDocPtr  doc;
    xmlParserCtxtPtr context;
    struct expression * root;

    context = xmlNewParserCtxt();
    doc = xmlCtxtReadFile(context, file_name, NULL, XML_PARSE_HUGE);
    assert(doc);


    root = create_expression(COMPILATION_UNIT, process_network(doc), process_variables_and_probabilities(doc));

    xmlFreeDoc(doc);

    return root;
}