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
    CELER_VALIDATE(env_var, "Environment variable G4LEDATA is not defined.");
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
AtomicRelaxationReader::operator()(int atomic_number) const
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
        CELER_VALIDATE(infile, "Couldn't open '" << filename << "'");

        int       des       = 0;
        real_type tr_energy = 0.;
        real_type tr_prob   = 0.;
        infile >> des >> des >> des;
        CELER_ASSERT(infile);
        result.designators.push_back(des);
        result.fluor.emplace_back();
        while (infile >> des >> tr_prob >> tr_energy)
        {
            CELER_ASSERT(infile);

            // End of shell data
            if (des == -1)
            {
                infile >> des >> tr_prob >> tr_energy;
                CELER_ASSERT(infile);

                // End of file
                if (des == -2)
                    break;

                // Designator for next shell data
                result.designators.push_back(des);
                result.fluor.emplace_back();
            }
            else
            {
                auto& shell = result.fluor.back();
                shell.initial_shell.push_back(des);
                shell.transition_prob.push_back(tr_prob);
                shell.transition_energy.push_back(tr_energy);
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
        CELER_VALIDATE(infile, "Couldn't open '" << filename << "'");

        int       des       = 0;
        int       auger_des = 0;
        real_type tr_energy = 0.;
        real_type tr_prob   = 0.;
        infile >> des >> des >> des >> des;
        CELER_ASSERT(infile);
        CELER_ASSERT(size_type(des) == result.designators.front());
        result.auger.emplace_back();
        while (infile >> des >> auger_des >> tr_prob >> tr_energy)
        {
            CELER_ASSERT(infile);

            // End of shell data
            if (des == -1)
            {
                infile >> des >> auger_des >> tr_prob >> tr_energy;
                CELER_ASSERT(infile);

                // End of file
                if (des == -2)
                    break;

                // Designator for next shell data
                CELER_ASSERT(size_type(des)
                             == result.designators[result.auger.size()]);
                result.auger.emplace_back();
            }
            else
            {
                auto& shell = result.auger.back();
                shell.initial_shell.push_back(des);
                shell.auger_shell.push_back(auger_des);
                shell.transition_prob.push_back(tr_prob);
                shell.transition_energy.push_back(tr_energy);
            }
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
