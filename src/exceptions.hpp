#ifndef HPGMP_EXCEPTIONS_HPP
#define HPGMP_EXCEPTIONS_HPP

#include <string>
#include <stdexcept>

class HandleNotCreatedError : public std::runtime_error
{
public:
    HandleNotCreatedError(const std::string& msg)
        : std::runtime_error("! Error: Handle could not be created: " + msg)
    {
    }
};

class HostDeviceCopyFailedError : public std::runtime_error
{
public:
    HostDeviceCopyFailedError(const std::string& msg)
        : std::runtime_error("! Error: could not copy " + msg)
    {
    }
};

class DeviceMemoryError : public std::runtime_error
{
public:
    DeviceMemoryError(const std::string& msg)
        : std::runtime_error("! Device memory error: " + msg)
    {
    }
};

class DeviceAPIError : public std::runtime_error
{
    static const std::string platform;
public:
    DeviceAPIError(const std::string& msg)
        : std::runtime_error(std::string("! ") + platform + " API " + msg + " failed!")
    {
    }
};

class MPICommError : public std::runtime_error
{
public:
    MPICommError(const std::string& msg)
        : std::runtime_error("! MPI: " + msg)
    {
    }
};

#endif
