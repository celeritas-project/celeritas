//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/demo-loop.cc
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterface.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/EnvironmentIO.json.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/io/EventReader.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/phys/PrimaryGenerator.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"
#include "celeritas/user/RootStepWriter.hh"
#include "celeritas/user/StepCollector.hh"
#include "celeritas/user/StepData.hh"

#include "LDemoIO.hh"
#include "Transporter.hh"
#include "Transporter.json.hh"

using std::cerr;
using std::cout;
using std::endl;
using namespace demo_loop;
using namespace celeritas;

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 */
void run(std::istream* is, std::shared_ptr<OutputRegistry> output)
{
    ScopedMem record_mem("demo_loop.run");
    // Read input options
    auto inp = nlohmann::json::parse(*is);

    if (inp.contains("cuda_heap_size"))
    {
        int heapSize = inp.at("cuda_heap_size").get<int>();
        celeritas::set_cuda_heap_size(heapSize);
    }
    if (inp.contains("cuda_stack_size"))
    {
        celeritas::set_cuda_stack_size(inp.at("cuda_stack_size").get<int>());
    }
    if (inp.contains("environ"))
    {
        // Specify env variables
        inp["environ"].get_to(celeritas::environment());
    }

    // For now, only do a single run
    auto run_args = inp.get<LDemoArgs>();
    output->insert(std::make_shared<OutputInterfaceAdapter<LDemoArgs>>(
        OutputInterface::Category::input,
        "*",
        std::make_shared<LDemoArgs>(run_args)));
    CELER_EXPECT(run_args);

    // Start timer for overall execution
    Stopwatch get_setup_time;

    // Construct core parameters
    auto core_params = build_core_params(run_args, output);

    // Initialize RootFileManager and store input data if requested
    std::shared_ptr<RootFileManager> root_manager;
    std::shared_ptr<StepCollector> step_collector;
    StepCollector::VecInterface step_interfaces;
    if (!run_args.mctruth_filename.empty())
    {
        root_manager = std::make_shared<RootFileManager>(
            run_args.mctruth_filename.c_str());

        // Store input and CoreParams data
        write_to_root(run_args, root_manager.get());
        write_to_root(*core_params, root_manager.get());

        // Create root step writer
        step_interfaces.push_back(std::make_shared<RootStepWriter>(
            root_manager,
            core_params->particle(),
            StepSelection::all(),
            make_write_filter(run_args.mctruth_filter)));
    }

    if (!step_interfaces.empty())
    {
        step_collector
            = std::make_unique<StepCollector>(std::move(step_interfaces),
                                              core_params->geometry(),
                                              core_params->max_streams(),
                                              core_params->action_reg().get());
    }

    // Load all the problem data and create transporter
    auto transport_ptr = build_transporter(run_args, core_params);
    double const setup_time = get_setup_time();

    // Run all the primaries
    TransporterResult result;
    std::vector<Primary> primaries;
    if (run_args.primary_gen_options)
    {
        std::mt19937 rng;
        auto generate_event = PrimaryGenerator<std::mt19937>::from_options(
            core_params->particle(), run_args.primary_gen_options);
        auto event = generate_event(rng);
        while (!event.empty())
        {
            primaries.insert(primaries.end(), event.begin(), event.end());
            event = generate_event(rng);
        }
    }
    else
    {
        EventReader read_event(run_args.hepmc3_filename.c_str(),
                               core_params->particle());
        auto event = read_event();
        while (!event.empty())
        {
            primaries.insert(primaries.end(), event.begin(), event.end());
            event = read_event();
        }
    }

    // Transport
    result = (*transport_ptr)(make_span(primaries));

    result.time.setup = setup_time;

    // TODO: convert individual results into OutputInterface so we don't have
    // to use this ugly "global" hack
    output->insert(OutputInterfaceAdapter<TransporterResult>::from_rvalue_ref(
        OutputInterface::Category::result, "*", std::move(result)));
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])
{
    ScopedRootErrorHandler scoped_root_error;
    ScopedMpiInit scoped_mpi(&argc, &argv);

    MpiCommunicator comm = [] {
        if (ScopedMpiInit::status() == ScopedMpiInit::Status::disabled)
            return MpiCommunicator{};

        return MpiCommunicator::comm_world();
    }();

    if (comm.size() > 1)
    {
        CELER_LOG(critical) << "TODO: this app cannot run in parallel";
        return EXIT_FAILURE;
    }

    // Process input arguments
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() != 2 || args[1] == "--help" || args[1] == "-h")
    {
        std::cerr << "usage: " << args[0] << " {input}.json" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize GPU
    celeritas::activate_device(celeritas::make_device(comm));

    std::string filename = args[1];
    std::ifstream infile;
    std::istream* instream = nullptr;
    if (filename == "-")
    {
        instream = &std::cin;
        filename = "<stdin>";  // For nicer output on failure
    }
    else
    {
        // Open the specified file
        infile.open(filename);
        if (!infile)
        {
            CELER_LOG(critical) << "Failed to open '" << filename << "'";
            return EXIT_FAILURE;
        }
        instream = &infile;
    }

    // Set up output
    auto output = std::make_shared<OutputRegistry>();

    int return_code = EXIT_SUCCESS;
    try
    {
        run(instream, output);
    }
    catch (std::exception const& e)
    {
        CELER_LOG(critical)
            << "While running input at " << filename << ": " << e.what();
        return_code = EXIT_FAILURE;
        output->insert(
            std::make_shared<ExceptionOutput>(std::current_exception()));
    }

    // Write system properties and (if available) results
    CELER_LOG(status) << "Saving output";
    output->output(&cout);
    cout << endl;

    return return_code;
}
