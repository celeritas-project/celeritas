//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantMaterialPropertyGetter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <G4Material.hh>

#include "corecel/inp/Grid.hh"
#include "celeritas/io/ImportUnits.hh"

#include "GeantProcessImporter.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Retrieve and store optical material properties, if present.
 *
 * Construct with a pointer to the material properties table (may be null).
 * All accessors return false if the table is null.
 */
class GeantMaterialPropertyGetter
{
  public:
    using MPT = G4MaterialPropertiesTable;

    //! Construct with a (possibly null) properties table pointer
    explicit GeantMaterialPropertyGetter(MPT const* mpt) : mpt_{mpt} {}

    //! True if the properties table is non-null
    explicit operator bool() const { return mpt_ != nullptr; }

    //! Get property for a single double
    bool operator()(double& dst, char const* name, ImportUnits q)
    {
        if (!*this || !mpt_->ConstPropertyExists(name))
        {
            return false;
        }
        dst = mpt_->GetConstProperty(name) * native_value_from_clhep(q);
        return true;
    }

    //! Get property for a single double (indexed component)
    bool operator()(double& dst, std::string name, int comp, ImportUnits q)
    {
        if (!*this)
        {
            return false;
        }
        // Geant4 10.6 and earlier require a const char* argument
        name += std::to_string(comp);
        return (*this)(dst, name.c_str(), q);
    }

    //! Get property for a physics vector
    bool
    operator()(inp::Grid& dst, std::string const& name, Array<ImportUnits, 2> q)
    {
        if (!*this)
        {
            return false;
        }
        // Geant4@10.7: G4MaterialPropertiesTable.GetProperty is not const
        // and <=10.6 require const char*
        auto const* g4vector
            = const_cast<MPT*>(mpt_)->GetProperty(name.c_str());
        if (!g4vector)
        {
            return false;
        }
        dst = import_physics_vector(*g4vector, q);
        return true;
    }

  private:
    MPT const* mpt_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
