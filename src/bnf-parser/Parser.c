/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 1 "Parser.y" /* yacc.c:339  */

#include "Parser.h"
#include "Lexer.h"
#include "expression.h"
#include "../constants.h"

#define YYDEBUG 1

int yyerror(YYLTYPE * yyltype, struct expression ** expression, yyscan_t scanner, const char *msg){
	fprintf(stderr, "Error on line: %d and column: %d: %s\n", yyltype->last_line, yyltype->last_column, msg);
	return 0;
}


#line 81 "Parser.c" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "Parser.h".  */
#ifndef YY_YY_PARSER_H_INCLUDED
# define YY_YY_PARSER_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif
/* "%code requires" blocks.  */
#line 16 "Parser.y" /* yacc.c:355  */


#ifndef YY_TYPEDEF_YY_SCANNER_T
#define YY_TYPEDEF_YY_SCANNER_T
typedef void* yyscan_t;
#endif


#line 120 "Parser.c" /* yacc.c:355  */

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    TOKEN_NETWORK = 258,
    TOKEN_VARIABLE = 259,
    TOKEN_PROBABILITY = 260,
    TOKEN_VARIABLETYPE = 261,
    TOKEN_DISCRETE = 262,
    TOKEN_DEFAULTVALUE = 263,
    TOKEN_TABLEVALUES = 264,
    TOKEN_L_CURLY_BRACE = 265,
    TOKEN_R_CURLY_BRACE = 266,
    TOKEN_L_BRACKET = 267,
    TOKEN_R_BRACKET = 268,
    TOKEN_L_PARENS = 269,
    TOKEN_R_PARENS = 270,
    TOKEN_SEMICOLON = 271,
    TOKEN_DECIMAL_LITERAL = 272,
    TOKEN_FLOATING_POINT_LITERAL = 273,
    TOKEN_WORD = 274,
    TOKEN_PROPERTY = 275
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 34 "Parser.y" /* yacc.c:355  */

	char word[50];
	int int_value;
	float float_value;
	struct expression * expression;

#line 160 "Parser.c" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif



int yyparse (struct expression ** expression, yyscan_t scanner);

#endif /* !YY_YY_PARSER_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 190 "Parser.c" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL \
             && defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE) + sizeof (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  7
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   79

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  21
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  25
/* YYNRULES -- Number of rules.  */
#define YYNRULES  55
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  87

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   275

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint8 yyrline[] =
{
       0,    87,    87,    90,    94,    98,   105,   106,   110,   111,
     114,   121,   122,   123,   124,   125,   128,   134,   135,   138,
     139,   140,   141,   144,   150,   154,   158,   162,   168,   171,
     174,   178,   182,   186,   192,   193,   196,   197,   198,   199,
     200,   201,   202,   203,   206,   209,   212,   215,   219,   224,
     228,   235,   238,   242,   246,   252
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "TOKEN_NETWORK", "TOKEN_VARIABLE",
  "TOKEN_PROBABILITY", "TOKEN_VARIABLETYPE", "TOKEN_DISCRETE",
  "TOKEN_DEFAULTVALUE", "TOKEN_TABLEVALUES", "TOKEN_L_CURLY_BRACE",
  "TOKEN_R_CURLY_BRACE", "TOKEN_L_BRACKET", "TOKEN_R_BRACKET",
  "TOKEN_L_PARENS", "TOKEN_R_PARENS", "TOKEN_SEMICOLON",
  "TOKEN_DECIMAL_LITERAL", "TOKEN_FLOATING_POINT_LITERAL", "TOKEN_WORD",
  "TOKEN_PROPERTY", "$accept", "input", "expr", "compilation_unit",
  "network_declaration", "network_content", "property_list", "property",
  "variable_or_probability_declaration", "variable_declaration",
  "variable_content", "property_or_variable_discrete", "variable_discrete",
  "variable_values_list", "probability_declaration",
  "probability_variables_list", "probability_names_list",
  "probability_content", "probability_content_entries",
  "probability_default_entry", "probability_entry",
  "probability_values_list", "probability_values", "probability_table",
  "floating_point_list", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275
};
# endif

#define YYPACT_NINF -34

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-34)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int8 yypact[] =
{
       4,     1,    29,   -34,   -34,     7,    36,   -34,    16,    49,
       7,   -34,   -34,    -7,   -34,    54,    15,    55,   -34,   -34,
     -34,   -34,    -5,   -34,    -3,   -34,   -34,   -34,    25,    13,
     -34,   -34,   -34,    59,   -34,   -34,    -1,   -34,   -34,   -34,
     -34,    44,    44,   -34,    40,    47,    17,   -34,   -34,    44,
     -34,    27,   -34,   -34,   -34,   -34,   -34,    32,    35,   -34,
     -34,    28,   -34,    47,   -34,   -34,   -34,    38,    51,   -34,
     -34,   -34,   -34,   -34,   -34,   -34,   -34,    56,    60,    41,
     -34,   -34,    19,    57,   -34,   -34,   -34
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     2,     3,    15,     0,     1,     0,     0,
       4,    11,    12,     0,     5,     0,     0,     0,    13,    14,
       6,    10,     0,     8,     0,    16,    31,    30,     0,     0,
      28,     7,     9,     0,    17,    19,     0,    20,    29,    33,
      32,     0,     0,    34,     0,    36,     0,    38,    40,     0,
      42,     0,    18,    21,    22,    53,    52,     0,     0,    48,
      47,     0,    35,    37,    39,    41,    43,     0,     0,    44,
      55,    54,    51,    46,    50,    49,    45,     0,     0,     0,
      25,    24,     0,     0,    27,    26,    23
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -34,   -34,   -34,   -34,   -34,   -34,   -28,   -22,   -34,    61,
     -34,   -34,    39,   -34,    62,   -34,   -34,   -34,   -34,    30,
      31,   -34,   -34,    33,   -33
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,     2,     3,     4,     5,    14,    22,    23,    10,    11,
      25,    36,    37,    82,    12,    17,    28,    30,    46,    47,
      48,    49,    61,    50,    57
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      32,    45,    35,    33,    20,    33,    31,     1,    34,    58,
      52,     8,     9,    21,    53,    21,    67,    21,    63,    21,
       6,    41,    42,    32,    43,    41,    42,    44,    62,     7,
      83,    44,    26,    21,    27,    15,    84,    21,    85,    68,
      38,    32,    39,    73,    40,    74,    13,    75,    69,    70,
      71,    72,    70,    71,    76,    70,    71,    59,    80,    60,
      81,    55,    56,    16,    24,    29,    51,    21,    77,    78,
      79,    18,    19,    86,     0,    54,    64,    65,     0,    66
};

static const yytype_int8 yycheck[] =
{
      22,    29,    24,     6,    11,     6,    11,     3,    11,    42,
      11,     4,     5,    20,    36,    20,    49,    20,    46,    20,
      19,     8,     9,    45,    11,     8,     9,    14,    11,     0,
      11,    14,    17,    20,    19,    19,    17,    20,    19,    12,
      15,    63,    17,    15,    19,    17,    10,    19,    16,    17,
      18,    16,    17,    18,    16,    17,    18,    17,    17,    19,
      19,    17,    18,    14,    10,    10,     7,    20,    17,    13,
      10,    10,    10,    16,    -1,    36,    46,    46,    -1,    46
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,    22,    23,    24,    25,    19,     0,     4,     5,
      29,    30,    35,    10,    26,    19,    14,    36,    30,    35,
      11,    20,    27,    28,    10,    31,    17,    19,    37,    10,
      38,    11,    28,     6,    11,    28,    32,    33,    15,    17,
      19,     8,     9,    11,    14,    27,    39,    40,    41,    42,
      44,     7,    11,    28,    33,    17,    18,    45,    45,    17,
      19,    43,    11,    27,    40,    41,    44,    45,    12,    16,
      17,    18,    16,    15,    17,    19,    16,    17,    13,    10,
      17,    19,    34,    11,    17,    19,    16
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    21,    22,    23,    24,    25,    26,    26,    27,    27,
      28,    29,    29,    29,    29,    29,    30,    31,    31,    32,
      32,    32,    32,    33,    34,    34,    34,    34,    35,    36,
      37,    37,    37,    37,    38,    38,    39,    39,    39,    39,
      39,    39,    39,    39,    40,    41,    42,    43,    43,    43,
      43,    44,    45,    45,    45,    45
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     2,     3,     2,     3,     1,     2,
       1,     1,     1,     2,     2,     0,     3,     2,     3,     1,
       1,     2,     2,     9,     1,     1,     2,     2,     3,     3,
       1,     1,     2,     2,     2,     3,     1,     2,     1,     2,
       1,     2,     1,     2,     3,     3,     3,     1,     1,     2,
       2,     3,     1,     1,     2,     2
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (&yylloc, expression, scanner, YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static unsigned
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  unsigned res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
 }

#  define YY_LOCATION_PRINT(File, Loc)          \
  yy_location_print_ (File, &(Loc))

# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value, Location, expression, scanner); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, struct expression ** expression, yyscan_t scanner)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  YYUSE (yylocationp);
  YYUSE (expression);
  YYUSE (scanner);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, struct expression ** expression, yyscan_t scanner)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  YY_LOCATION_PRINT (yyoutput, *yylocationp);
  YYFPRINTF (yyoutput, ": ");
  yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp, expression, scanner);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule, struct expression ** expression, yyscan_t scanner)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                       , &(yylsp[(yyi + 1) - (yynrhs)])                       , expression, scanner);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule, expression, scanner); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that float-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp, struct expression ** expression, yyscan_t scanner)
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);
  YYUSE (expression);
  YYUSE (scanner);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/*----------.
| yyparse.  |
`----------*/

int
yyparse (struct expression ** expression, yyscan_t scanner)
{
/* The lookahead symbol.  */
int yychar;


/* The semantic value of the lookahead symbol.  */
/* Default value used for initialization, for pacifying older GCCs
   or non-GCC compilers.  */
YY_INITIAL_VALUE (static YYSTYPE yyval_default;)
YYSTYPE yylval YY_INITIAL_VALUE (= yyval_default);

/* Location data for the lookahead symbol.  */
static YYLTYPE yyloc_default
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
YYLTYPE yylloc = yyloc_default;

    /* Number of syntax errors so far.  */
    int yynerrs;

    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.
       'yyls': related to locations.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    /* The location stack.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls;
    YYLTYPE *yylsp;

    /* The locations where the error started and ended.  */
    YYLTYPE yyerror_range[3];

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yylsp = yyls = yylsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  yylsp[0] = yylloc;
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yyls1, yysize * sizeof (*yylsp),
                    &yystacksize);

        yyls = yyls1;
        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex (&yylval, &yylloc, scanner);
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location.  */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 87 "Parser.y" /* yacc.c:1646  */
    { *expression = (yyvsp[0].expression); }
#line 1435 "Parser.c" /* yacc.c:1646  */
    break;

  case 3:
#line 90 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = (yyvsp[0].expression); }
#line 1441 "Parser.c" /* yacc.c:1646  */
    break;

  case 4:
#line 94 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(COMPILATION_UNIT, (yyvsp[-1].expression), (yyvsp[0].expression)); }
#line 1447 "Parser.c" /* yacc.c:1646  */
    break;

  case 5:
#line 98 "Parser.y" /* yacc.c:1646  */
    { struct expression * network_expr = create_expression(NETWORK_DECLARATION, (yyvsp[0].expression), NULL);
												strncpy(network_expr->value, (yyvsp[-1].word), CHAR_BUFFER_SIZE);
												(yyval.expression) =  network_expr;
												}
#line 1456 "Parser.c" /* yacc.c:1646  */
    break;

  case 6:
#line 105 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression( NETWORK_CONTENT, NULL, NULL); }
#line 1462 "Parser.c" /* yacc.c:1646  */
    break;

  case 7:
#line 106 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression( NETWORK_CONTENT, (yyvsp[-1].expression), NULL ); }
#line 1468 "Parser.c" /* yacc.c:1646  */
    break;

  case 8:
#line 110 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROPERTY_LIST, (yyvsp[0].expression), NULL); }
#line 1474 "Parser.c" /* yacc.c:1646  */
    break;

  case 9:
#line 111 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROPERTY_LIST, (yyvsp[-1].expression), (yyvsp[0].expression)); }
#line 1480 "Parser.c" /* yacc.c:1646  */
    break;

  case 10:
#line 114 "Parser.y" /* yacc.c:1646  */
    {
							struct expression * property_expression =  create_expression(PROPERTY, NULL, NULL);
							strncpy(property_expression->value, (yyvsp[0].word), CHAR_BUFFER_SIZE);
							(yyval.expression) =  property_expression;
						}
#line 1490 "Parser.c" /* yacc.c:1646  */
    break;

  case 11:
#line 121 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, (yyvsp[0].expression), NULL); }
#line 1496 "Parser.c" /* yacc.c:1646  */
    break;

  case 12:
#line 122 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, (yyvsp[0].expression), NULL); }
#line 1502 "Parser.c" /* yacc.c:1646  */
    break;

  case 13:
#line 123 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, (yyvsp[-1].expression), (yyvsp[0].expression)); }
#line 1508 "Parser.c" /* yacc.c:1646  */
    break;

  case 14:
#line 124 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, (yyvsp[-1].expression), (yyvsp[0].expression)); }
#line 1514 "Parser.c" /* yacc.c:1646  */
    break;

  case 15:
#line 125 "Parser.y" /* yacc.c:1646  */
    { }
#line 1520 "Parser.c" /* yacc.c:1646  */
    break;

  case 16:
#line 128 "Parser.y" /* yacc.c:1646  */
    { struct expression * expr = create_expression(VARIABLE_DECLARATION, (yyvsp[0].expression), NULL);
													strncpy(expr->value, (yyvsp[-1].word), CHAR_BUFFER_SIZE);
													(yyval.expression) = expr; 
													}
#line 1529 "Parser.c" /* yacc.c:1646  */
    break;

  case 17:
#line 134 "Parser.y" /* yacc.c:1646  */
    {}
#line 1535 "Parser.c" /* yacc.c:1646  */
    break;

  case 18:
#line 135 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(VARIABLE_CONTENT, (yyvsp[-1].expression), NULL);  }
#line 1541 "Parser.c" /* yacc.c:1646  */
    break;

  case 19:
#line 138 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(VARIABLE_OR_PROBABILITY, (yyvsp[0].expression), NULL); }
#line 1547 "Parser.c" /* yacc.c:1646  */
    break;

  case 20:
#line 139 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(VARIABLE_OR_PROBABILITY, (yyvsp[0].expression), NULL); }
#line 1553 "Parser.c" /* yacc.c:1646  */
    break;

  case 21:
#line 140 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(VARIABLE_OR_PROBABILITY, (yyvsp[-1].expression), (yyvsp[0].expression)); }
#line 1559 "Parser.c" /* yacc.c:1646  */
    break;

  case 22:
#line 141 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(VARIABLE_OR_PROBABILITY, (yyvsp[-1].expression), (yyvsp[0].expression)); }
#line 1565 "Parser.c" /* yacc.c:1646  */
    break;

  case 23:
#line 144 "Parser.y" /* yacc.c:1646  */
    {
																																								struct expression * variable_discrete = create_expression(VARIABLE_DISCRETE, (yyvsp[-2].expression), NULL);
																																								variable_discrete->int_value = (yyvsp[-5].int_value);
																																								(yyval.expression) = variable_discrete;
																																							}
#line 1575 "Parser.c" /* yacc.c:1646  */
    break;

  case 24:
#line 150 "Parser.y" /* yacc.c:1646  */
    { struct expression * values_list = create_expression(VARIABLE_VALUES_LIST, NULL, NULL);
					strncpy(values_list->value, (yyvsp[0].word), CHAR_BUFFER_SIZE);
					(yyval.expression) = values_list;
					}
#line 1584 "Parser.c" /* yacc.c:1646  */
    break;

  case 25:
#line 154 "Parser.y" /* yacc.c:1646  */
    {struct expression * values_list = create_expression(VARIABLE_VALUES_LIST, NULL, NULL);
                             					snprintf(values_list->value, CHAR_BUFFER_SIZE, "%d", (yyvsp[0].int_value));
                             					(yyval.expression) = values_list;
	                        }
#line 1593 "Parser.c" /* yacc.c:1646  */
    break;

  case 26:
#line 158 "Parser.y" /* yacc.c:1646  */
    { struct expression * values_list = create_expression(VARIABLE_VALUES_LIST, (yyvsp[-1].expression), NULL);
										strncpy(values_list->value, (yyvsp[0].word), CHAR_BUFFER_SIZE);
										(yyval.expression) = values_list;
										}
#line 1602 "Parser.c" /* yacc.c:1646  */
    break;

  case 27:
#line 162 "Parser.y" /* yacc.c:1646  */
    {struct expression * values_list = create_expression(VARIABLE_VALUES_LIST, (yyvsp[-1].expression), NULL);
                                                                              					snprintf(values_list->value, CHAR_BUFFER_SIZE, "%d", (yyvsp[0].int_value));
                                                                              					(yyval.expression) = values_list;
                                                 	                        }
#line 1611 "Parser.c" /* yacc.c:1646  */
    break;

  case 28:
#line 168 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_DECLARATION, (yyvsp[-1].expression), (yyvsp[0].expression)); }
#line 1617 "Parser.c" /* yacc.c:1646  */
    break;

  case 29:
#line 171 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_VARIABLES_LIST, (yyvsp[-1].expression), NULL); }
#line 1623 "Parser.c" /* yacc.c:1646  */
    break;

  case 30:
#line 174 "Parser.y" /* yacc.c:1646  */
    { struct expression * names_list = create_expression(PROBABILITY_VARIABLE_NAMES, NULL, NULL);
				   strncpy(names_list->value, (yyvsp[0].word), CHAR_BUFFER_SIZE);
				   (yyval.expression) = names_list;
				   }
#line 1632 "Parser.c" /* yacc.c:1646  */
    break;

  case 31:
#line 178 "Parser.y" /* yacc.c:1646  */
    {struct expression * names_list = create_expression(PROBABILITY_VARIABLE_NAMES, NULL, NULL);
                             snprintf(names_list->value, CHAR_BUFFER_SIZE, "%d", (yyvsp[0].int_value));
                            (yyval.expression) = names_list;
	                        }
#line 1641 "Parser.c" /* yacc.c:1646  */
    break;

  case 32:
#line 182 "Parser.y" /* yacc.c:1646  */
    {struct expression * names_list = create_expression(PROBABILITY_VARIABLE_NAMES, (yyvsp[-1].expression), NULL);
										   strncpy(names_list->value, (yyvsp[0].word), CHAR_BUFFER_SIZE);
										   (yyval.expression) = names_list;
										}
#line 1650 "Parser.c" /* yacc.c:1646  */
    break;

  case 33:
#line 186 "Parser.y" /* yacc.c:1646  */
    {struct expression * names_list = create_expression(PROBABILITY_VARIABLE_NAMES, (yyvsp[-1].expression), NULL);
                                                                                snprintf(names_list->value, CHAR_BUFFER_SIZE, "%d", (yyvsp[0].int_value));
                                                                               (yyval.expression) = names_list;
                                                   	                        }
#line 1659 "Parser.c" /* yacc.c:1646  */
    break;

  case 34:
#line 192 "Parser.y" /* yacc.c:1646  */
    {}
#line 1665 "Parser.c" /* yacc.c:1646  */
    break;

  case 35:
#line 193 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_CONTENT, (yyvsp[-1].expression), NULL); }
#line 1671 "Parser.c" /* yacc.c:1646  */
    break;

  case 36:
#line 196 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_CONTENT_LIST, (yyvsp[0].expression), NULL); }
#line 1677 "Parser.c" /* yacc.c:1646  */
    break;

  case 37:
#line 197 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_CONTENT_LIST, (yyvsp[-1].expression), (yyvsp[0].expression)); }
#line 1683 "Parser.c" /* yacc.c:1646  */
    break;

  case 38:
#line 198 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_CONTENT_LIST, (yyvsp[0].expression), NULL); }
#line 1689 "Parser.c" /* yacc.c:1646  */
    break;

  case 39:
#line 199 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_CONTENT_LIST, (yyvsp[-1].expression), (yyvsp[0].expression)); }
#line 1695 "Parser.c" /* yacc.c:1646  */
    break;

  case 40:
#line 200 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_CONTENT_LIST, (yyvsp[0].expression), NULL); }
#line 1701 "Parser.c" /* yacc.c:1646  */
    break;

  case 41:
#line 201 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_CONTENT_LIST, (yyvsp[-1].expression), (yyvsp[0].expression)); }
#line 1707 "Parser.c" /* yacc.c:1646  */
    break;

  case 42:
#line 202 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_CONTENT_LIST, (yyvsp[0].expression), NULL); }
#line 1713 "Parser.c" /* yacc.c:1646  */
    break;

  case 43:
#line 203 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_CONTENT_LIST, (yyvsp[-1].expression), (yyvsp[0].expression)); }
#line 1719 "Parser.c" /* yacc.c:1646  */
    break;

  case 44:
#line 206 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_DEFAULT_ENTRY, (yyvsp[-1].expression), NULL); }
#line 1725 "Parser.c" /* yacc.c:1646  */
    break;

  case 45:
#line 209 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_ENTRY, (yyvsp[-2].expression), (yyvsp[-1].expression)); }
#line 1731 "Parser.c" /* yacc.c:1646  */
    break;

  case 46:
#line 212 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_VALUES_LIST, (yyvsp[-1].expression), NULL); }
#line 1737 "Parser.c" /* yacc.c:1646  */
    break;

  case 47:
#line 215 "Parser.y" /* yacc.c:1646  */
    { struct expression * values_list = create_expression(PROBABILITY_VALUES, NULL, NULL); 
					strncpy(values_list->value, (yyvsp[0].word), CHAR_BUFFER_SIZE); 
					(yyval.expression) = values_list; 
					}
#line 1746 "Parser.c" /* yacc.c:1646  */
    break;

  case 48:
#line 219 "Parser.y" /* yacc.c:1646  */
    {
                            struct expression * values_list = create_expression(PROBABILITY_VALUES, NULL, NULL);
                            					snprintf(values_list->value, CHAR_BUFFER_SIZE, "%d", (yyvsp[0].int_value));
                            					(yyval.expression) = values_list;
    					}
#line 1756 "Parser.c" /* yacc.c:1646  */
    break;

  case 49:
#line 224 "Parser.y" /* yacc.c:1646  */
    { struct expression * values_list = create_expression(PROBABILITY_VALUES, (yyvsp[-1].expression), NULL);
										strncpy(values_list->value, (yyvsp[0].word), CHAR_BUFFER_SIZE);
										(yyval.expression) = values_list;
										}
#line 1765 "Parser.c" /* yacc.c:1646  */
    break;

  case 50:
#line 228 "Parser.y" /* yacc.c:1646  */
    {
                                                                           struct expression * values_list = create_expression(PROBABILITY_VALUES, (yyvsp[-1].expression), NULL);
                                                                           					snprintf(values_list->value, CHAR_BUFFER_SIZE, "%d", (yyvsp[0].int_value));
                                                                           					(yyval.expression) = values_list;
                                                   					}
#line 1775 "Parser.c" /* yacc.c:1646  */
    break;

  case 51:
#line 235 "Parser.y" /* yacc.c:1646  */
    { (yyval.expression) = create_expression(PROBABILITY_TABLE, (yyvsp[-1].expression), NULL); }
#line 1781 "Parser.c" /* yacc.c:1646  */
    break;

  case 52:
#line 238 "Parser.y" /* yacc.c:1646  */
    { struct expression * fp_list = create_expression(FLOATING_POINT_LIST, NULL, NULL);
									 fp_list->float_value = (yyvsp[0].float_value);
									 (yyval.expression) = fp_list;
									}
#line 1790 "Parser.c" /* yacc.c:1646  */
    break;

  case 53:
#line 242 "Parser.y" /* yacc.c:1646  */
    {struct expression * fp_list = create_expression(FLOATING_POINT_LIST, NULL, NULL);
                                     									 fp_list->float_value = (float)(yyvsp[0].int_value);
                                     									 (yyval.expression) = fp_list;
	                                }
#line 1799 "Parser.c" /* yacc.c:1646  */
    break;

  case 54:
#line 246 "Parser.y" /* yacc.c:1646  */
    {
	                                                    struct expression * fp_list = create_expression(FLOATING_POINT_LIST, (yyvsp[-1].expression), NULL);
														 fp_list->float_value = (yyvsp[0].float_value);
														 (yyval.expression) = fp_list;
														}
#line 1809 "Parser.c" /* yacc.c:1646  */
    break;

  case 55:
#line 252 "Parser.y" /* yacc.c:1646  */
    {
                                                            struct expression * fp_list = create_expression(FLOATING_POINT_LIST, (yyvsp[-1].expression), NULL);
                                                             fp_list->float_value = (float)(yyvsp[0].int_value);
                                                             (yyval.expression) = fp_list;
                                                            }
#line 1819 "Parser.c" /* yacc.c:1646  */
    break;


#line 1823 "Parser.c" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (&yylloc, expression, scanner, YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (&yylloc, expression, scanner, yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }

  yyerror_range[1] = yylloc;

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc, expression, scanner);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  yyerror_range[1] = yylsp[1-yylen];
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp, yylsp, expression, scanner);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  /* Using YYLLOC is tempting, but would change the location of
     the lookahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT (yyloc, yyerror_range, 2);
  *++yylsp = yyloc;

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (&yylloc, expression, scanner, YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc, expression, scanner);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp, yylsp, expression, scanner);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
