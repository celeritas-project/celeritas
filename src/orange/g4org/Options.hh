//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Options.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "geocel/detail/LengthUnits.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace g4org
{
//! How to inline volumes used only once
enum class InlineSingletons
{
    none,  //!< Never
    untransformed,  //!< Only if not translated nor rotated
    unrotated,  //!< Only if translated
    all,  //!< Always
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Construction options for Geant4 conversion.
 */
struct Options
{
    //!@{
    //! \name Problem scale and tolerance

    //! Scale factor (input unit length), customizable for unit testing
    double unit_length{celeritas::lengthunits::millimeter};
    //! Construction and tracking tolerance (native units)
    Tolerance<> tol;

    //!@}
    //!@{
    //! \name Structural conversion

    //! Volumes with up to this many children construct an explicit interior
    unsigned int explicit_interior_threshold{2};

    //! Forcibly inline logical volumes that are only used once
    InlineSingletons inline_singletons{InlineSingletons::untransformed};

    //! Forcibly copy child volumes that have union boundaries
    bool inline_unions{true};

    //!@}
    //!@{
    //! \name Debug output

    //! Write output about volumes being converted
    bool verbose_volumes{false};
    //! Write output about proto-universes being constructed
    bool verbose_structure{false};

    //! Write interpreted geometry to a JSON file
    std::string proto_output_file;
    //! Write intermediate debug output (CSG construction) to a JSON file
    std::string debug_output_file;

    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
