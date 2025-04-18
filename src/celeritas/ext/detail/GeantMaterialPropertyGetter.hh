//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantMaterialPropertyGetter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <G4Material.hh>

#include "celeritas/inp/Grid.hh"
#include "celeritas/io/ImportUnits.hh"

#include "GeantProcessImporter.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Retrieve and store optical material properties, if present.
 */
struct GeantMaterialPropertyGetter
{
    using MPT = G4MaterialPropertiesTable;

    MPT const& mpt;

    //! Get property for a single double
    bool operator()(double* dst, char const* name, ImportUnits q)
    {
        if (!mpt.ConstPropertyExists(name))
        {
            return false;
        }
        *dst = mpt.GetConstProperty(name) * native_value_from_clhep(q);
        return true;
    }

    //! Get property for a single double
    bool operator()(double* dst, std::string name, int comp, ImportUnits q)
    {
        // Geant4 10.6 and earlier require a const char* argument
        name += std::to_string(comp);
        return (*this)(dst, name.c_str(), q);
    }

    //! Get property for a physics vector
    bool
    operator()(inp::Grid* dst, std::string const& name, Array<ImportUnits, 2> q)
    {
        // Geant4@10.7: G4MaterialPropertiesTable.GetProperty is not const
        // and <=10.6 require const char*
        auto const* g4vector = const_cast<MPT&>(mpt).GetProperty(name.c_str());
        if (!g4vector)
        {
            return false;
        }
        *dst = import_physics_vector(*g4vector, q);
        return true;
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
