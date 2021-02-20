#pragma once
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <string>

#define ERROR_OUTPUT(T) std::cout << "\033[0;31m" << T << "\033[0m" << std::endl
#define SUCCESS_OUTPUT(T) std::cout << "\033[0;32m" << T << "\033[0m" << std::endl
#define OTHER_OUTPUT(T) std::cout << "\033[0;33m" << T << "\033[0m" << std::endl
#define OPEN_GREEN std::cout << "\033[0;32m"
#define CLOSE_COLOR std::cout << "\033[0m"
namespace mySLAM
{
    class config
    {
    private:
        config()
        {
            assert(!fileName.empty());
            fileStorage = cv::FileStorage(fileName, cv::FileStorage::READ);
        }
        cv::FileStorage fileStorage;
        static std::string fileName;

    public:
        static std::shared_ptr<config> getInstance()
        {
            static std::shared_ptr<config> ptr;
            if (ptr == NULL)
            {
                ptr = std::shared_ptr<config>(new config());
            }
            return ptr;
        }
        static void setFileName(const std::string &path)
        {
            fileName = path;
        }
        template <typename T>
        T getData(const char *name) const
        {
            return static_cast<T>(fileStorage[name]);
        }
    };

} // namespace mySLAM