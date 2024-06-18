//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Converter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include "celeritas_config.h"
#include "orange/OrangeInput.hh"
#include "orange/OrangeTypes.hh"

//---------------------------------------------------------------------------//
// Forward declarations
//---------------------------------------------------------------------------//

class G4LogicalVolume;
class G4VPhysicalVolume;

namespace celeritas
{
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
 * Return the new world volume and a mapping of Geant4 logical volumes to
 * VecGeom-based volume IDs.
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
    using arg_type = G4VPhysicalVolume const*;
    using MapLvVolId = std::unordered_map<G4LogicalVolume const*, VolumeId>;
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
        //! Write intermediate debug ouput (CSG construction) to a JSON file
        std::string debug_output_file;
    };

    struct result_type
    {
        OrangeInput input;
        MapLvVolId volumes;  //! TODO
    };

  public:
    // Construct with options
    explicit Converter(Options&&);

    //! Construct with default options
    Converter() : Converter{Options{}} {}

    // Convert the world
    result_type operator()(arg_type);

  private:
    Options opts_;
};

//---------------------------------------------------------------------------//

#if !(CELERITAS_USE_GEANT4 \
      && CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
inline Converter::Converter(Options&&)
{
    CELER_DISCARD(opts_);
}

inline auto Converter::operator()(arg_type) -> result_type
{
    CELER_NOT_CONFIGURED("Geant4 with double-precision real_type");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
