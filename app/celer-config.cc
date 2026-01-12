//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-config.cc
//! \brief Write the Celeritas JSON configuration to stdout
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <iostream>
#include <string>
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/AssertIO.json.hh"  // IWYU pragma keep
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceIO.json.hh"  // IWYU pragma keep
#include "corecel/sys/ScopedMpiInit.hh"

#include "CliUtils.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
struct Args
{
    int indent{1};
    bool show_device{false};
};

//---------------------------------------------------------------------------//
nlohmann::json device_to_json()
{
    if (Device::num_devices() == 0)
    {
        CELER_LOG(info) << "No GPUs were detected";
        return nullptr;
    }

    try
    {
        celeritas::activate_device();
        CELER_VALIDATE(celeritas::Device::num_devices() != 0,
                       << "no GPUs were detected");
    }
    catch (...)
    {
        return output_to_json(ExceptionOutput{std::current_exception()});
    }

    return celeritas::device();
}

void run(Args const& args)
{
    CELER_VALIDATE(args.indent >= -1 && args.indent < 80,
                   << "invalid indentation " << args.indent);

    auto result = output_to_json(BuildOutput{});
    if (args.show_device)
    {
        result["device"] = device_to_json();
    }

    std::cout << result.dump(/* indent = */ args.indent) << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace
}  // namespace app
}  // namespace celeritas

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])
{
    using namespace celeritas::app;

    celeritas::ScopedMpiInit scoped_mpi(&argc, &argv);
    if (scoped_mpi.is_world_multiprocess())
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    auto& cli = cli_app();
    cli.description("Write the Celeritas build configuration to stdout");

    Args args;
    cli.add_flag("-d,--device", args.show_device, "Activate and query GPU");
    cli.add_option("-i,--indent", args.indent, "JSON indentation")
        ->default_val(args.indent);

    CELER_CLI11_PARSE(argc, argv);
    return run_safely(run, args);
}
