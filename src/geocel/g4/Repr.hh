//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/Repr.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4ThreeVector.hh>

#include "corecel/io/ScopedStreamFormat.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct G4ThreeVectorRepr
{
    G4ThreeVector const& obj;

    friend std::ostream& operator<<(std::ostream& os,
                                    G4ThreeVectorRepr const& value)
    {
        ScopedStreamFormat save_fmt(&os);
        os.precision(15);
        return os << '{' << value.obj[0] << ", " << value.obj[1] << ", "
                  << value.obj[2] << '}';
    }
};

inline auto repr(G4ThreeVector const& vec)
{
    return G4ThreeVectorRepr{vec};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
