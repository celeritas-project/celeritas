//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptionsMessenger.cc
//---------------------------------------------------------------------------//
#include "SetupOptionsMessenger.hh"

#include <type_traits>
#include <G4UIcmdWithABool.hh>
#include <G4UIcmdWithAnInteger.hh>
#include <G4Version.hh>

#include "corecel/Assert.hh"
#include "corecel/sys/Device.hh"

#include "SetupOptions.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Helper traits for dealing with Geant4 UI commands
template<class T, class = void>
struct UICommandTraits;

template<>
struct UICommandTraits<std::string>
{
    static constexpr char type_info = 's';
    static std::string const& to_string(G4String const& v) { return v; }
    static std::string from_string(G4String const& v) { return v; }
};

template<>
struct UICommandTraits<bool>
{
    static constexpr char type_info = 'b';
    static G4String to_string(bool v)
    {
        return G4UIcommand::ConvertToString(v);
    }
    static bool from_string(G4String const& v)
    {
        return G4UIcommand::ConvertToBool(v.c_str());
    }
};

template<class T>
struct UICommandTraits<T, std::enable_if_t<std::is_integral_v<T>>>
{
    static constexpr char type_info = 'i';
    static std::string to_string(T v) { return std::to_string(v); }
    static long from_string(G4String const& v)
    {
        // Conversion to long int introduced in Geant4 10.7.0
#if G4VERSION_NUMBER >= 1070
        return G4UIcommand::ConvertToLongInt(v.c_str());
#else
        G4long vl;
        std::istringstream is(v);
        is >> vl;
        return vl;
#endif
    }
};

template<>
struct UICommandTraits<double>
{
    static constexpr char type_info = 'd';
    static G4String to_string(double v)
    {
        return G4UIcommand::ConvertToString(v);
    }
    static double from_string(G4String const& v)
    {
        return G4UIcommand::ConvertToDouble(v.c_str());
    }
};

//---------------------------------------------------------------------------//
//! Object-oriented command to invoke a user macro
class CelerCommand : public G4UIcommand
{
  public:
    using G4UIcommand::G4UIcommand;

    virtual void apply(G4String const& newValue) const = 0;
    virtual G4String get() const = 0;
};

template<class T>
class CelerParamCommand final : public CelerCommand
{
  private:
    using CmdTraits = UICommandTraits<T>;

  public:
    CelerParamCommand(G4UIdirectory const& parent,
                      char const* cmd_path,
                      G4UImessenger* mess,
                      T* dest)
        : CelerCommand(
              (parent.GetCommandPath() + std::string(cmd_path)).c_str(), mess)
        , dest_(dest)
    {
        // NOTE: Geant4 takes ownership of the parameter
        auto param = std::make_unique<G4UIparameter>(CmdTraits::type_info);
        // Set default value based on the current pointed-to value
        param->SetDefaultValue(CmdTraits::to_string(*dest).c_str());
        // Save to this command
        this->SetParameter(param.release());

        // We're for setup only
        this->AvailableForStates(G4State_PreInit, G4State_Init);
    }

    void apply(G4String const& value_str) const final
    {
        auto converted = CmdTraits::from_string(value_str);

        //! \todo Add validation for non-matching types, i.e. int to unsigned
        *this->dest_ = static_cast<T>(converted);
    }

    G4String get() const final { return CmdTraits::to_string(*this->dest_); }

  private:
    T* dest_;
};

template<class T>
CelerParamCommand(G4UIdirectory const&, char const*, G4UImessenger*, T*)
    -> CelerParamCommand<T>;

//---------------------------------------------------------------------------//
//! Helper class for constructing a "directory"
class CelerDirectory final : public G4UIdirectory
{
  public:
    CelerDirectory(char const* path, char const* desc) : G4UIdirectory(path)
    {
        this->SetGuidance(desc);
    }
};

//---------------------------------------------------------------------------//
}  // namespace

SetupOptionsMessenger::SetupOptionsMessenger(SetupOptions* options)
{
    CELER_EXPECT(options);

    auto add_cmd = [this](auto* ptr, char const* relpath, char const* desc) {
        CELER_ASSERT(!directories_.empty());
        this->commands_.emplace_back(
            new CelerParamCommand{*directories_.back(), relpath, this, ptr});
        this->commands_.back()->SetGuidance(desc);
    };

    directories_.emplace_back(
        new CelerDirectory("/celer/", "Celeritas setup options"));
    add_cmd(&options->geometry_file,
            "geometryFile",
            "Override detector geometry with a custom GDML");
    add_cmd(&options->output_file,
            "outputFile",
            "Filename for JSON diagnostic output");
    add_cmd(&options->physics_output_file,
            "physicsOutputFile",
            "Filename for ROOT dump of physics data");
    add_cmd(&options->offload_output_file,
            "offloadOutputFile",
            "Filename for HepMC3/ROOT dump of offloaded tracks");
    add_cmd(&options->geometry_output_file,
            "geometryOutputFile",
            "Filename for GDML export");
    add_cmd(&options->max_num_tracks,
            "maxNumTracks",
            "Number of track \"slots\" to be transported simultaneously");
    add_cmd(&options->max_num_events,
            "maxNumEvents",
            "Maximum number of events in use (DEPRECATED)");
    add_cmd(&options->max_steps,
            "maxNumSteps",
            "Limit on number of step iterations before aborting");
    add_cmd(&options->initializer_capacity,
            "maxInitializers",
            "Maximum number of track initializers (primaries+secondaries)");
    add_cmd(&options->secondary_stack_factor,
            "secondaryStackFactor",
            "At least the average number of secondaries per track slot");
    add_cmd(&options->auto_flush,
            "autoFlush",
            "Number of tracks to buffer before offloading");
    add_cmd(&options->max_field_substeps,
            "maxFieldSubsteps",
            "Limit on substeps in the field propagator");

    directories_.emplace_back(new CelerDirectory(
        "/celer/detector/", "Celeritas sensitive detector setup options"));
    add_cmd(&options->sd.enabled,
            "enabled",
            "Call back to Geant4 sensitive detectors");

    if (Device::num_devices() > 0)
    {
        directories_.emplace_back(new CelerDirectory(
            "/celer/cuda/", "Celeritas CUDA setup options"));
        add_cmd(&options->cuda_stack_size,
                "stackSize",
                "Set the CUDA per-thread stack size for VecGeom");
        add_cmd(&options->cuda_heap_size,
                "heapSize",
                "Set the CUDA per-thread heap size for VecGeom");
        add_cmd(&options->action_times,
                "actionTimes",
                "Add timers around every action (may reduce performance)");
        add_cmd(&options->default_stream,
                "defaultStream",
                "Launch all kernels on the default stream (REMOVED)");
    }

    add_cmd(&options->slot_diagnostic_prefix,
            "slotDiagnosticPrefix",
            "Filename base for slot diagnostics");
}

//---------------------------------------------------------------------------//
//! Default destructor
SetupOptionsMessenger::~SetupOptionsMessenger() = default;

//---------------------------------------------------------------------------//
//! Dispatch a command
void SetupOptionsMessenger::SetNewValue(G4UIcommand* cmd, G4String val)
{
    auto* celer_cmd = dynamic_cast<CelerCommand*>(cmd);
    CELER_EXPECT(celer_cmd);

    celer_cmd->apply(val);
}

//---------------------------------------------------------------------------//
//! Get the value of the given command
G4String SetupOptionsMessenger::GetCurrentValue(G4UIcommand* cmd)
{
    auto* celer_cmd = dynamic_cast<CelerCommand*>(cmd);
    return celer_cmd->get();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
