//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/Repr.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4ThreeVector.hh>

#include "corecel/io/Repr.hh"

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
        os.precision(14);
        return os << '{' << value.obj[0] << ", " << value.obj[1] << ", "
                  << value.obj[2] << '}';
    }
};

inline G4ThreeVectorRepr repr(G4ThreeVector const& vec)
{
    return {vec};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
