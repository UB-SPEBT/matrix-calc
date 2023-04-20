#include "cmd_args.hh"

namespace po = boost::program_options;
// namespace cls = boost::program_options::command_line_style;
void print_usage_msg() {
  std::cout << "\nBasic Usage:\n  " << BINNAME
            << " -c <path> -i <num> -d <num>\n";
};
// Print the help message of the program
void print_help_msg(po::options_description desc) {
  print_usage_msg();
  std::cout << desc;
};

// Command-line arguments parser
int cmd_args_parser(int argc, char **argv, po::variables_map &vm,
                    std::string &configDir) {
  po::options_description generic("Options", 1024, 512);
  auto configOption = po::value<std::string>(&configDir);

  configOption->required();
  configOption->value_name("path");
  generic.add_options()("help,h", "produce help message")(
      "config,c", configOption, "Set the config file path");
  generic.add_options()("img-zIndex,i", po::value<int>()->required(),
                        "image space axial direction index")(
      "det-zIndex,d", po::value<int>()->required(),
      "detector space axial direction index");
  generic.add_options()("rotation,r", po::value<float>()->default_value(0),
                        "panel rotation in degrees");

  try {

    po::parsed_options const intermediate =
        po::parse_command_line(argc, argv, generic);
    // po::variables_map vm;

    po::store(intermediate, vm);

    if (vm.count("help")) {
      print_help_msg(generic);
      return 1;
    }

    po::notify(vm);
  } catch (std::exception &e) {
    // std::cerr << "Error: " << e.what() << "\n";
    std::cout << "Error: Missing required arguments.\n";
    // print_usage_msg();
    std::cout << "Run "
              << "'" << BINNAME << " -h' for more information.\n";
    return 2;
  } catch (...) {
    std::cerr << "Unknown error!"
              << "\n";
    return 9;
  }

  return 0;
}