//
//
#include <libxml/parser.h>
#include <libxml/xpath.h>

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "../constants.h"
#include "xml-expression.h"

#if LIBXML_VERSION < 20901
/**
 * xmlXPathSetContextNode:
 * @node: the node to to use as the context node
 * @ctx:  the XPath context
 *
 * Sets 'node' as the context node. The node must be in the same
 * document as that associated with the context.
 *
 * Returns -1 in case of error or 0 if successful
 */
int
xmlXPathSetContextNode(xmlNodePtr node, xmlXPathContextPtr ctx) {
    if ((node == NULL) || (ctx == NULL))
        return(-1);

    if (node->doc == ctx->doc) {
        ctx->node = node;
        return(0);
    }
    return(-1);
}



/**
 * xmlXPathNodeEval:
 * @node: the node to to use as the context node
 * @str:  the XPath expression
 * @ctx:  the XPath context
 *
 * Evaluate the XPath Location Path in the given context. The node 'node'
 * is set as the context node. The context node is not restored.
 *
 * Returns the xmlXPathObjectPtr resulting from the evaluation or NULL.
 *         the caller has to free the object.
 */
xmlXPathObjectPtr
xmlXPathNodeEval(xmlNodePtr node, const xmlChar *str, xmlXPathContextPtr ctx) {
    if (str == NULL)
        return(NULL);
    if (xmlXPathSetContextNode(node, ctx) < 0)
        return(NULL);
    return(xmlXPathEval(str, ctx));
}


#endif

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

static unsigned int count_number_of_nodes(xmlDocPtr doc){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;

    unsigned int num_nodes;

    result = get_node_set(doc, (xmlChar *)"//NETWORK/VARIABLE");
    assert(result);

    node_set = result->nodesetval;

    assert(node_set->nodeNr >= 0);
    num_nodes = (unsigned int)node_set->nodeNr;

    xmlXPathFreeObject(result);
    return num_nodes;
}

static unsigned int count_number_of_edges(xmlDocPtr doc){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    unsigned int num_edges;

    result = get_node_set(doc, (xmlChar *)"//NETWORK/DEFINITION/GIVEN");
    assert(result);

    node_set = result->nodesetval;

    assert(node_set->nodeNr >= 0);
    num_edges = (unsigned int)node_set->nodeNr;

    xmlXPathFreeObject(result);
    return num_edges;
}

static unsigned int add_variables_to_graph(xmlDocPtr doc, xmlNodePtr node, Graph_t graph, unsigned int node_index){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    unsigned int num_variables, char_index, num_vertices;
    int i;
    xmlChar * variable_name;
    char * node_name;

    num_variables = 0;
    num_vertices = graph->current_num_vertices;

    result = get_subnode_set(doc, (xmlChar *)".//OUTCOME/text()", node);
    assert(result);

    node_set = result->nodesetval;
    for(i = 0; i < node_set->nodeNr; ++i){
        char_index = num_vertices * MAX_STATES * CHAR_BUFFER_SIZE + i * CHAR_BUFFER_SIZE;
        variable_name = xmlNodeListGetString(doc, node_set->nodeTab[i], 0);
        node_name = &graph->variable_names[char_index];

        strncpy(node_name, (char *)variable_name, CHAR_BUFFER_SIZE);
        xmlFree(variable_name);

        num_variables++;
    }


    xmlXPathFreeObject(result);

    return num_variables;
}

static void add_node_to_graph(xmlDocPtr doc, xmlNodePtr node, Graph_t graph, unsigned int node_index){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    unsigned int num_variables;
    char buffer[CHAR_BUFFER_SIZE];
    xmlChar *value;

    num_variables = add_variables_to_graph(doc, node, graph, node_index);

    result = get_subnode_set(doc, (xmlChar *)".//NAME/text()", node);
    assert(result);
    node_set = result->nodesetval;
    assert(node_set->nodeNr == 1);

    value = xmlNodeListGetString(doc, node_set->nodeTab[0], 0);
    strncpy(buffer, (char *)value, CHAR_BUFFER_SIZE);

    xmlFree(value);
    xmlXPathFreeObject(result);

    graph_add_node(graph, num_variables, buffer);
}

static void add_nodes_to_graph(xmlDocPtr doc, Graph_t graph){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    int i;

    result = get_node_set(doc, (xmlChar *)"//NETWORK/VARIABLE");
    assert(result);

    node_set = result->nodesetval;

    for(i = 0; i < node_set->nodeNr; ++i){
        add_node_to_graph(doc, node_set->nodeTab[i], graph, (unsigned int)i);
    }

    xmlXPathFreeObject(result);
}

static unsigned int count_probabilities(xmlDocPtr doc, xmlNodePtr definition){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    unsigned int count;
    xmlChar * value;
    char * split;

    count = 0;

    result = get_subnode_set(doc, (xmlChar *)".//TABLE/text()", definition);
    assert(result);

    node_set = result->nodesetval;
    assert(node_set->nodeNr > 0);

    value = xmlNodeListGetString(doc, node_set->nodeTab[node_set->nodeNr - 1], 0);
    split = strtok((char *)value, " \t\n\r");
    while(split != NULL){
        count++;
        split = strtok(NULL, " \t\n\r");
    }

    xmlFree(value);
    xmlXPathFreeObject(result);

    return count;
}

static void build_probabilities(xmlDocPtr doc, xmlNodePtr definition, struct belief *belief, unsigned int length){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    xmlChar * value;
    char * split;
    unsigned int i;

    i = 0;

    result = get_subnode_set(doc, (xmlChar *)".//TABLE/text()", definition);
    assert(result);

    node_set = result->nodesetval;
    assert(node_set->nodeNr > 0);

    value = xmlNodeListGetString(doc, node_set->nodeTab[node_set->nodeNr - 1], 0);
    split = strtok((char *)value, " \t\n\r");
    while(split != NULL && i < length){
        if(strlen(split) > 0){
            sscanf(split, "%f", &belief->data[i]);
            i++;
        }
        split = strtok(NULL, " \t\n\r");
    }

    xmlFree(value);
    xmlXPathFreeObject(result);
}

static void fill_in_for_node_name(xmlDocPtr doc, xmlNodePtr definition, char * buffer){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    xmlChar * value;

    result = get_subnode_set(doc, (xmlChar *)".//FOR/text()", definition);
    assert(result);
    node_set = result->nodesetval;
    assert(node_set->nodeNr == 1);

    value = xmlNodeListGetString(doc, node_set->nodeTab[0], 0);
    strncpy(buffer, (char *)value, CHAR_BUFFER_SIZE);

    xmlFree(value);
    xmlXPathFreeObject(result);
}


static void add_observed_node_to_graph(xmlDocPtr doc, xmlNodePtr definition, Graph_t graph){
    xmlXPathObjectPtr result;
    char dest_node_name[CHAR_BUFFER_SIZE];
    struct belief belief;
    unsigned int num_probabilities;
    unsigned int dest_node_index;
    unsigned int i;

    // check if edge or observed node
    result = get_subnode_set(doc, (xmlChar *)".//GIVEN/text()", definition);
    if(result == NULL){
        for(i = 0; i < MAX_STATES; ++i){
            belief.data[i] = 0.0f;
        }

        num_probabilities = count_probabilities(doc, definition);
        assert(num_probabilities < MAX_STATES);
        build_probabilities(doc, definition, &belief, num_probabilities);

        fill_in_for_node_name(doc, definition, dest_node_name);
        dest_node_index = find_node_by_name(dest_node_name, graph);

        graph_set_node_state(graph, dest_node_index, num_probabilities, &belief);
    }
    else{
        xmlXPathFreeObject(result);
    }
}

static void reverse_probability_table(float * probability_table, int num_probabilities){
    int i;
    float temp;

    for(i = 0; i < num_probabilities/2; ++i){
        temp = probability_table[i];
        probability_table[i] = probability_table[num_probabilities - i - 1];
        probability_table[num_probabilities - i - 1] = temp;
    }
}

static void add_edges_to_graph(xmlDocPtr doc, xmlNodePtr definition, Graph_t graph){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    char dest_node_name[CHAR_BUFFER_SIZE];
    char src_node_name[CHAR_BUFFER_SIZE];
    struct belief *new_belief;
    unsigned int num_probabilities;
    unsigned int j, k, offset, slice, index, delta, next, diff, dest_index, src_index;
    int i;
    xmlChar * value;

    struct joint_probability sub_graph;
    struct joint_probability transpose;

    // check if edge or observed node
    result = get_subnode_set(doc, (xmlChar *)".//GIVEN/text()", definition);
    if(result == NULL){
        return;
    }
    assert(result);
    node_set = result->nodesetval;

    num_probabilities = count_probabilities(doc, definition);
    new_belief = (struct belief *)malloc(sizeof(struct belief));
    assert(new_belief);
    build_probabilities(doc, definition, new_belief, num_probabilities);

    fill_in_for_node_name(doc, definition, dest_node_name);
    dest_index = find_node_by_name(dest_node_name, graph);
    //assert(dest_index >= 0);
    //assert(dest_index < graph->current_num_vertices);
    slice = num_probabilities / graph->node_num_vars[dest_index];

    offset = 1;
    for(i = node_set->nodeNr - 1; i >= 0; --i){
        value = xmlNodeListGetString(doc, node_set->nodeTab[i], 0);
        strncpy(src_node_name, (char *)value, CHAR_BUFFER_SIZE);
        xmlFree(value);

        src_index = find_node_by_name(src_node_name, graph);
        //printf("Adding edge: (%s)->(%s)\n", src_node_name, dest_node_name);
        //printf("indices: (%d)->(%d)\n", src_index, dest_index);
        //assert(src_index >= 0);
        //assert(src_index < graph->current_num_vertices);

        delta = graph->node_num_vars[src_index];

        for(k = 0; k < graph->node_num_vars[dest_index]; ++k){
            for(j = 0; j < graph->node_num_vars[src_index]; ++j){
                sub_graph.data[j][k] = 0.0;
                transpose.data[k][j] = 0.0;
            }
        }

        for(k = 0; k < graph->node_num_vars[dest_index]; ++k){
            for(j = 0; j < graph->node_num_vars[src_index]; ++j){
                diff = 0;
                index = j * offset + diff;
                while(index <= slice) {
                    index = j * offset + diff;
                    next = (j + 1) * offset + diff;
                    //printf("Current Index: %d; Next: %d; Delta: %d; Diff: %d\n", index, next, delta, diff);
                    while (index < next) {
                        sub_graph.data[j][k] += new_belief->data[index + k * slice];
                        index++;
                    }
                    index += delta * offset;
                    diff += delta * offset;
                }
            }
        }

        for(j = 0; j < graph->node_num_vars[src_index]; ++j){
            for(k = 0; k < graph->node_num_vars[dest_index]; ++k){
                transpose.data[k][j] = sub_graph.data[j][k];
            }
        }

        graph_add_edge(graph, src_index, dest_index, graph->node_num_vars[src_index], graph->node_num_vars[dest_index], &sub_graph);
        if(graph->observed_nodes[src_index] != 1 ){
            graph_add_edge(graph, dest_index, src_index, graph->node_num_vars[dest_index], graph->node_num_vars[src_index], &transpose);
        }


        offset *= graph->node_num_vars[src_index];
    }

    free(new_belief);
    xmlXPathFreeObject(result);
}

static void add_definitions_to_graph(xmlDocPtr doc, Graph_t graph){
    xmlXPathObjectPtr result;
    xmlNodeSetPtr node_set;
    int i;

    result = get_node_set(doc, (xmlChar *)"//NETWORK/DEFINITION");
    assert(result);

    node_set = result->nodesetval;

    for(i = 0; i < node_set->nodeNr; ++i){
        add_observed_node_to_graph(doc, node_set->nodeTab[i], graph);
    }
    for(i = 0; i < node_set->nodeNr; ++i){
        add_edges_to_graph(doc, node_set->nodeTab[i], graph);
    }

    xmlXPathFreeObject(result);
}

Graph_t parse_xml_file(const char * file_name){
    xmlDocPtr  doc;
    xmlParserCtxtPtr context;
    int file_access;
    unsigned int num_nodes, num_edges;
    Graph_t graph;

    // ensure file path exists
    file_access = access(file_name, F_OK);
    assert( file_access != -1 );

    context = xmlNewParserCtxt();
    doc = xmlCtxtReadFile(context, file_name, NULL, XML_PARSE_HUGE);
    assert(doc);

    num_nodes = count_number_of_nodes(doc);
    num_edges = count_number_of_edges(doc);

    graph = create_graph(num_nodes, num_edges * 2);
    add_nodes_to_graph(doc, graph);
    add_definitions_to_graph(doc, graph);

    xmlFreeDoc(doc);

    return graph;
}
