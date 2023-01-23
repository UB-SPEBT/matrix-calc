#include <fstream>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include <cstdio>

int parse_config_json(std::string configFilePath, rapidjson::Document &document)
{
    using namespace rapidjson;

    char readBuffer[65536];
    try
    {
        FILE *inFile_ptr;
        inFile_ptr = fopen(configFilePath.c_str(), "rb");
        FileReadStream inStream(inFile_ptr, readBuffer, sizeof(readBuffer));

        // RJ_document.Parse<kParseCommentsFlag>("\"test/*.zip//comment\"");

        if (document.ParseStream<kParseCommentsFlag>(inStream).HasParseError())
        {
            fprintf(stderr, "\nJSON file parsing error (@offset %u): %s\n",
                    (unsigned)document.GetErrorOffset(),
                    GetParseError_En(document.GetParseError()));
        }

        fclose(inFile_ptr);
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error!"
                  << "\n";
        return 9;
    }
    return 0;
}