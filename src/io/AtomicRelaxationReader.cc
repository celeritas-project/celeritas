//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AtomicRelaxationReader.cc
//---------------------------------------------------------------------------//
#include "AtomicRelaxationReader.hh"

#include <fstream>
#include <sstream>
#include "base/SoftEqual.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the reader using the G4LEDATA environment variable to get the path
 * to the data.
 */
AtomicRelaxationReader::AtomicRelaxationReader()
{
    const char* env_var = std::getenv("G4LEDATA");
    CELER_VALIDATE(env_var,
                   << "environment variable G4LEDATA is not defined (needed "
                      "to locate atomic relaxation data)");
    {
        std::ostringstream os;
        os << env_var << "/fluor";
        fluor_path_ = os.str();
    }
    {
        std::ostringstream os;
        os << env_var << "/auger";
        auger_path_ = os.str();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct the reader with the path to the directory containing the data.
 */
AtomicRelaxationReader::AtomicRelaxationReader(const char* fluor_path,
                                               const char* auger_path)
    : fluor_path_(fluor_path), auger_path_(auger_path)
{
    CELER_EXPECT(!fluor_path_.empty());
    CELER_EXPECT(!auger_path_.empty());

    if (fluor_path_.back() == '/')
        fluor_path_.pop_back();
    if (auger_path_.back() == '/')
        auger_path_.pop_back();
}

//---------------------------------------------------------------------------//
/*!
 * Read the data for the given element.
 */
AtomicRelaxationReader::result_type
AtomicRelaxationReader::operator()(AtomicNumber atomic_number) const
{
    CELER_EXPECT(atomic_number > 0 && atomic_number < 101);

    // EADL does not provide transition data for Z < 6
    result_type result;
    if (atomic_number < 6)
    {
        return result;
    }

    std::string Z = std::to_string(atomic_number);

    // Read fluorescence transition probabilities and subshell designators. All
    // lines in the data file are 3 columns. The first line of each section is
    // the subshell designator of the initial vacancy. Each of the following
    // rows contains data for a different transition, where the columns are:
    // designator of the shell of the electron that fills the vacancy,
    // transition probability, and transition energy. A row of -1 marks the end
    // of the shell data, and -2 marks the end of the file.
    {
        std::string   filename = fluor_path_ + "/fl-tr-pr-" + Z + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile,
                       << "failed to open '" << filename
                       << "' (should contain fluorescence data)");

        int    des    = 0;
        double energy = 0;
        double prob   = 0;

        // Get the designator for the first section of shell data
        infile >> des >> des >> des;
        CELER_ASSERT(infile);
        result.shells.emplace_back();
        result.shells.back().designator = des;
        while (infile >> des >> prob >> energy)
        {
            // End of shell data
            if (des == -1)
            {
                // Get the designator for the next section of shell data
                infile >> des >> des >> des;
                CELER_ASSERT(infile);

                // End of file
                if (des == -2)
                    break;

                result.shells.emplace_back();
                result.shells.back().designator = des;
            }
            else
            {
                result.shells.back().fluor.push_back({des, 0, prob, energy});
            }
        }
    }

    // Read Auger transition probabilities. All lines in the data file are 4
    // columns. The first line of each section is the subshell designator of
    // the initial vacancy. Each of the following rows contains data for a
    // different transition, where the columns are: designator of the shell of
    // the electron that fills the vacancy, designator of the Auger electron
    // shell, transition probability, and transition energy. A row of -1 marks
    // the end of the shell data, and -2 marks the end of the file.
    {
        std::string   filename = auger_path_ + "/au-tr-pr-" + Z + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile,
                       << "failed to open '" << filename
                       << "' (should contain Auger transition data)");

        int    des       = 0;
        int    auger_des = 0;
        double energy    = 0;
        double prob      = 0;

        // Get the designator for the first section of shell data
        infile >> des >> des >> des >> des;
        CELER_ASSERT(infile);
        auto shell = result.shells.begin();
        CELER_ASSERT(des == shell->designator);
        while (infile >> des >> auger_des >> prob >> energy)
        {
            // End of shell data
            if (des == -1)
            {
                // Get the designator for the next section of shell data
                infile >> des >> des >> des >> des;
                CELER_ASSERT(infile);

                // End of file
                if (des == -2)
                    break;

                ++shell;
                CELER_ASSERT(shell != result.shells.end());
                CELER_ASSERT(des == shell->designator);
            }
            else
            {
                shell->auger.push_back({des, auger_des, prob, energy});
            }
        }
    }

    // Renormalize the transition probabilities so that the sum over all
    // radiative and non-radiative transitions for a given subshell is 1
    for (auto& shell : result.shells)
    {
        double norm = 0;
        for (const auto& transition : shell.fluor)
            norm += transition.probability;
        for (const auto& transition : shell.auger)
            norm += transition.probability;
        CELER_ASSERT(soft_near(1., norm, 1.e-5));

        norm = 1. / norm;
        for (auto& transition : shell.fluor)
            transition.probability *= norm;
        for (auto& transition : shell.auger)
            transition.probability *= norm;
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
