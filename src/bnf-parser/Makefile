# Makefile

FILES = Lexer.c Parser.c expression.c main.c ../graph/graph.c ../edge/edge.c ../node/node.c
CC = gcc
CFLAGS = -g3

test: $(FILES)
	$(CC) $(CFLAGS) $(FILES) -o test

Lexer.c: Lexer.l
	flex Lexer.l

Parser.c: Parser.y Lexer.c
	bison Parser.y

clean:
	rm -rf *.o *~ Lexer.c Parser.c Parser.h test
