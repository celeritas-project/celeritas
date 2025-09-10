//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/run-offload.cc
//! \brief Minimal Geant4 application with Celeritas offloading
//---------------------------------------------------------------------------//
#include <filesystem>
#include <iostream>
#include <memory>
#include <FTFP_BERT.hh>
#include <G4RunManagerFactory.hh>
#include <G4UImanager.hh>
#include <accel/TrackingManagerConstructor.hh>
#include <accel/TrackingManagerIntegration.hh>
#include <corecel/io/BuildOutput.hh>
#include <corecel/io/Logger.hh>
#include <corecel/sys/TypeDemangler.hh>

#include "ActionInitialization.hh"
#include "BuildInfo.hh"
#include "DetectorConstruction.hh"
#include "MakeCelerOptions.hh"

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Return the shorter of the relative path to current directory or the full
 * path.
 */
std::string shorter_path(std::string const& path_str)
{
    namespace fs = std::filesystem;

    if (path_str.empty())
        return path_str;

    fs::path path{path_str};
    fs::path cwd = fs::current_path();

    try
    {
        std::string rel_path = fs::relative(path, cwd).string();
        return (rel_path.size() < path_str.size()) ? rel_path : path_str;
    }
    catch (fs::filesystem_error const&)
    {
        // If we can't get a relative path (e.g., different drives on Windows)
        return path_str;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print information about the example's build system and dependencies.
 *
 * This uses the CMake-exported build information stored in BuildInfo.hh
 * (configured by cmake to the build directory from the source directory's
 * BuildInfo.hh.in), as well as the configuration printed by \c
 * celeritas::BuildOutput.
 *
 * \note Because this function uses CELER_LOG, it must \em not be called before
 * Celeritas logging is initialized (during the first call to \c
 * celeritas::TrackingManagerIntegration::Instance ).
 */
void print_build_info(char const* argv0)
{
    namespace fs = std::filesystem;
    namespace ex = celeritas::example;

    fs::path cwd = fs::current_path();
    CELER_LOG(info) << "Working directory: " << cwd.string();
    CELER_LOG(debug) << "Executable: " << shorter_path(argv0);
    CELER_LOG(debug) << "Source code: " << shorter_path(ex::source_dir);
    CELER_LOG(debug) << "Build dir: " << shorter_path(ex::build_dir);
    CELER_LOG(debug) << "Celeritas install: "
                     << shorter_path(ex::celeritas_install_dir);
    CELER_LOG(debug) << "Geant4 install: "
                     << shorter_path(ex::geant4_install_dir);
    CELER_LOG(debug) << "Full Celeritas configuration: "
                     << celeritas::BuildOutput{};
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Geant4-Celeritas offloading template.
 *
 * See README for details.
 */
int main(int argc, char* argv[])
{
    if (argc > 2)
    {
        // Print help message
        std::cout << "Usage: " << argv[0] << " [input.mac]" << std::endl;
        return EXIT_FAILURE;
    }

    using namespace celeritas::example;

    std::unique_ptr<G4RunManager> run_manager{
        G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default)};

    // Initialize Celeritas
    auto& tmi = celeritas::TrackingManagerIntegration::Instance();

    // Print diagnostics about the build setup and chosen run manager
    print_build_info(argv[0]);
    CELER_LOG(info) << "Run manager type: "
                    << celeritas::TypeDemangler<G4RunManager>{}(*run_manager);

    // Initialize physics with celeritas offload
    auto* physics_list = new FTFP_BERT{/* verbosity = */ 0};
    physics_list->RegisterPhysics(
        new celeritas::TrackingManagerConstructor(&tmi));
    run_manager->SetUserInitialization(physics_list);
    tmi.SetOptions(MakeCelerOptions());

    // Initialize geometry and actions
    run_manager->SetUserInitialization(new DetectorConstruction());
    run_manager->SetUserInitialization(new ActionInitialization());

    if (argc == 1)
    {
        // Run four events
        run_manager->Initialize();
        run_manager->BeamOn(2);
    }
    else
    {
        // Run through the UI
        auto* ui = G4UImanager::GetUIpointer();
        ui->ApplyCommand("/control/execute " + std::string(argv[1]));
    }

    std::cout << "Done!" << std::endl;
    return EXIT_SUCCESS;
}
