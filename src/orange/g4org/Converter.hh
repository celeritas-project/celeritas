//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Converter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include "corecel/Config.hh"

#include "orange/OrangeInput.hh"
#include "orange/OrangeTypes.hh"

//---------------------------------------------------------------------------//
// Forward declarations
//---------------------------------------------------------------------------//

class G4LogicalVolume;

namespace celeritas
{
class GeantGeoParams;

struct OrangeInput;
namespace orangeinp
{
class ProtoInterface;
}

namespace g4org
{
//---------------------------------------------------------------------------//
/*!
 * Create an ORANGE geometry model from an in-memory Geant4 model.
 *
 * Return a complete geometry input, including a mapping of internal
 * ORANGE volume IDs to structural volume IDs.
 *
 * The default Geant4 "tolerance" (often used as surface "thickness") is 1e-9
 * mm, and the relative tolerance when specifying a length scale is 1e-11 (so
 * the default macro length scale is expected to be 100 mm = 10 cm).
 * That relative tolerance is *much* too small for any quadric operations or
 * angular rotations to be differentiated, so for now we'll stick with the
 * ORANGE default tolerance of 1e-8 relative, and we assume a 1mm length scale.
 */
class Converter
{
  public:
    //!@{
    //! \name Type aliases
    using arg_type = GeantGeoParams const&;
    //!@}

    //! Input options for the conversion
    struct Options
    {
        //! Write output about volumes being converted
        bool verbose{false};
        //! Manually specify a tracking/construction tolerance
        Tolerance<> tol;
        //! Write interpreted geometry to a JSON file
        std::string proto_output_file;
        //! Write intermediate debug output (CSG construction) to a JSON file
        std::string debug_output_file;
    };

    struct result_type
    {
        OrangeInput input;
    };

  public:
    // Construct with options
    explicit Converter(Options&&);

    //! Construct with default options
    Converter() : Converter{Options{}} {}

    // Convert the world
    result_type operator()(GeantGeoParams const&);

  private:
    Options opts_;
};

//---------------------------------------------------------------------------//

#if !CELERITAS_USE_GEANT4
inline Converter::Converter(Options&&)
{
    CELER_DISCARD(opts_);
}

inline auto Converter::operator()(arg_type) -> result_type
{
    CELER_NOT_CONFIGURED("Geant4");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
