#include <boost/program_options.hpp>
#include <boost/program_options/cmdline.hpp>

#define BINNAME "matcal"

namespace po = boost::program_options;
namespace cls = boost::program_options::command_line_style;

// Print the help message of the program
void print_usage_msg()
{
    std::cout << "\nUsage:\n\n  " << BINNAME << " -C [ --config ] <path-to-config-file>\n\n";
};
// Print the help message of the program
void print_help_msg(po::options_description desc)
{
    print_usage_msg();
    std::cout << desc;
};

// Command-line arguments parser
int cmd_args_parser(int argc, char **argv,
                    std::string &configDir)
{
    po::options_description desc("Options", 1024, 512);
    auto configOption = po::value<std::string>(&configDir);
    configOption->required();
    configOption->value_name("<path-to-config-file>");
    desc.add_options()("help,h", "produce help message")("config,C", configOption, "Set the config file path");

    try
    {

        po::parsed_options const intermediate = po::parse_command_line(argc, argv, desc);
        po::variables_map vm;

        po::store(intermediate, vm);

        if (vm.count("help"))
        {
            print_help_msg(desc);
            return 1;
        }

        po::notify(vm);
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage_msg();
        std::cout << "\nRun "
                  << "'" << BINNAME << " -h' for more information.\n";
        return 2;
    }
    catch (...)
    {
        std::cerr << "Unknown error!"
                  << "\n";
        return 9;
    }

    return 0;
}