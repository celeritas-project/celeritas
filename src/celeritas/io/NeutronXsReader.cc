//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/NeutronXsReader.cc
//---------------------------------------------------------------------------//
#include "NeutronXsReader.hh"

#include <fstream>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"

#include "ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the reader using the G4PARTICLEXSDATA environment variable to get
 * the path to the data.
 */
NeutronXsReader::NeutronXsReader(NeutronXsType type) : type_(type)
{
    std::string const& dir = celeritas::getenv("G4PARTICLEXSDATA");
    CELER_VALIDATE(!dir.empty(),
                   << "environment variable G4PARTICLEXSDATA is not defined "
                      "(needed to locate neutron cross section data)");
    path_ = dir + "/neutron";
}

//---------------------------------------------------------------------------//
/*!
 * Construct the reader with the path to the directory containing the data.
 */
NeutronXsReader::NeutronXsReader(NeutronXsType type, char const* path)
    : type_(type), path_(path)
{
    CELER_EXPECT(!path_.empty());
    if (path_.back() == '/')
    {
        path_.pop_back();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Read the data for the given element.
 */
NeutronXsReader::result_type
NeutronXsReader::operator()(AtomicNumber atomic_number) const
{
    CELER_EXPECT(atomic_number);

    std::string z_str = std::to_string(atomic_number.unchecked_get());
    CELER_LOG(debug) << "Reading neutron xs data for " << to_cstring(type_)
                     << " Z=" << z_str;

    result_type result;

    // Read neutron elastic cross section data for the given atomic_number
    {
        std::string filename = path_ + "/" + to_cstring(type_) + z_str;
        std::ifstream infile(filename);
        CELER_VALIDATE(infile,
                       << "failed to open '" << filename
                       << "' (should contain cross section data)");

        // Set the physics vector type
        result.vector_type = ImportPhysicsVectorType::free;

        // Read tabulated energies and cross sections
        double energy_min = 0.;
        double energy_max = 0.;
        size_type size = 0;
        infile >> energy_min >> energy_max >> size >> size;
        CELER_VALIDATE(size > 0,
                       << "incorrect neutron cross section size " << size);
        result.x.resize(size);
        result.y.resize(size);

        MmSqMicroXs input_xs;
        for (size_type i = 0; i < size; ++i)
        {
            CELER_ASSERT(infile);
            // Convert to the celeritas units::barn (units::BarnXs.value())
            // from clhep::mm^2 as stored in G4PARTICLEXS/neutron/el data
            infile >> result.x[i] >> input_xs.value();
            result.y[i]
                = native_value_to<units::BarnXs>(native_value_from(input_xs))
                      .value();
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Get the string value for a neutron cross section data type.
 */
char const* to_cstring(NeutronXsType value)
{
    static EnumStringMapper<NeutronXsType> const to_cstring_impl{
        "cap", "el", "inel"};
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
