#include <boost/program_options.hpp>
#include <boost/program_options/cmdline.hpp>
#include <iostream>
#include <map>

#define BINNAME "matcal"

namespace po = boost::program_options;
// namespace cls = boost::program_options::command_line_style;

// Print the help message of the program
void print_usage_msg();
int cmd_args_parser(int, char **, po::variables_map &, std::string &);