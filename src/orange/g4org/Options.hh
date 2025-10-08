//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Options.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string>

#include "corecel/Config.hh"

#include "geocel/detail/LengthUnits.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace g4org
{
//---------------------------------------------------------------------------//
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
 *
 * Note that most of these should \em never be touched when running an actual
 * problem. If the length unit is changed, the resulting geometry is
 * inconsistent with Geant4's scale.
 *
 * \warning Currently ORANGE tracking requires:
 * - inline unions to be true (see
 * https://github.com/celeritas-project/celeritas/issues/1260)
 * - remove_interior to be true (see
 * https://github.com/celeritas-project/celeritas/issues/2012 )
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

    //! Forcibly inline volumes that have no children
    bool inline_childless{true};

    //! Forcibly inline volumes that are only used once
    InlineSingletons inline_singletons{InlineSingletons::untransformed};

    //! Forcibly copy child volumes that have union boundaries
    bool inline_unions{true};

    //! Replace 'interior' unit boundaries with 'true' and simplify
    bool remove_interior{true};

    //! Use DeMorgan's law to replace "not all of" with "any of not"
    bool remove_negated_join{false};

    //!@}
    //!@{
    //! \name Debug output

    //! Write output about volumes being converted
    bool verbose_volumes{false};
    //! Write output about proto-universes being constructed
    bool verbose_structure{false};

    //! Write converted Geant4 object structure to a JSON file
    std::string objects_output_file;
    //! Write constructed CSG surfaces and tree to a JSON file
    std::string csg_output_file;
    //! Write final org.json to a JSON file
    std::string org_output_file;

    //!@}
};

//---------------------------------------------------------------------------//
// NOTE: these are implemented in OptionsIO.json.cc

char const* to_cstring(InlineSingletons value);

// Helper to read from a file or stream
std::istream& operator>>(std::istream& is, Options&);
// Helper to write to a file or stream
std::ostream& operator<<(std::ostream& os, Options const&);

#if !CELERITAS_USE_GEANT4
inline std::istream& operator>>(std::istream&, Options&)
{
    CELER_NOT_CONFIGURED("Geant4");
}
inline std::ostream& operator<<(std::ostream&, Options const&)
{
    CELER_NOT_CONFIGURED("Geant4");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
