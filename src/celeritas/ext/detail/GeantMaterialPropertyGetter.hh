//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantMaterialPropertyGetter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
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

    // Default constructor to allow deferred construction
    GeantMaterialPropertyGetter() = default;

    // Construct with a (possibly null) properties table pointer
    inline GeantMaterialPropertyGetter(MPT const* mpt, char const* desc);

    //! True if the properties table is non-null
    explicit operator bool() const { return mpt_ != nullptr; }

    // Get scalar property
    inline bool
    operator()(double& dst, std::string const& name, ImportUnits q) const;

    // Get physics vector property
    inline bool operator()(inp::Grid& dst,
                           std::string const& name,
                           Array<ImportUnits, 2> q) const;

    //! Get the string label
    char const* c_str() const
    {
        CELER_EXPECT(*this);
        return desc_;
    }

  private:
    MPT const* mpt_{nullptr};
    char const* desc_{nullptr};
};

// Write a description of the properties being queried
inline std::ostream&
operator<<(std::ostream&, GeantMaterialPropertyGetter const&);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a (possibly null) properties table pointer.
 *
 * If the table is not null, the description must not be null.
 */
GeantMaterialPropertyGetter::GeantMaterialPropertyGetter(MPT const* mpt,
                                                         char const* desc)
    : mpt_{mpt}, desc_{desc}
{
    CELER_EXPECT(desc_ || (mpt_ == nullptr));
}

//---------------------------------------------------------------------------//
/*!
 * Get property for a scalar.
 */
bool GeantMaterialPropertyGetter::operator()(double& dst,
                                             std::string const& name,
                                             ImportUnits q) const
{
    if (!*this || !mpt_->ConstPropertyExists(name.c_str()))
    {
        return false;
    }
    dst = mpt_->GetConstProperty(name.c_str()) * native_value_from_clhep(q);
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Get property for a physics vector.
 */
bool GeantMaterialPropertyGetter::operator()(inp::Grid& dst,
                                             std::string const& name,
                                             Array<ImportUnits, 2> q) const
{
    if (!*this)
    {
        return false;
    }
    // Geant4@10.7: G4MaterialPropertiesTable.GetProperty is not const
    // and <=10.6 require const char*
    auto const* g4vector = const_cast<MPT*>(mpt_)->GetProperty(name.c_str());
    if (!g4vector)
    {
        return false;
    }
    dst = import_physics_vector(*g4vector, q);
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Write description or "[INVALID]" if the properties table is null.
 */
std::ostream& operator<<(std::ostream& os, GeantMaterialPropertyGetter const& g)
{
    if (!g)
    {
        return os << "[INVALID]";
    }
    return os << g.c_str();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
