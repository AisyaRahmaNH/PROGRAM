#include <iostream>
using namespace std;

int main ()
{
    char operasi;
    double angka1, angka2, hasil;

    while (true)
    {
        // Tampilkan menu dan pengguna dapat memasukkan operasi
        cout << "Selamat Datang di Kalkulator!" << endl;
        cout << "Silakan masukkan operasi (+, -, *, /) atau 'Q' untuk Quit (keluar) : ";
        cin >> operasi;

        if (operasi == 0)
        {
            cout << "Perhitungan selesai. Selamat tinggal!" << endl;
            break;
        }

        // Masukkan angka
        cout << "Masukkan 2 angka : ";
        cin >> angka1 >> angka2;

        // Tampilkan di perhitungan
        switch (operasi)
        {
            case '+':
                hasil = angka1 + angka2;
                break;
            case '-':
                hasil = angka1 - angka2;
                break;
            case '*':
                hasil = angka1 * angka2;
            case '/':
                if (angka2 == 0)
                {
                    cout << "Error! Pembagian dengan angka 0 tidak diizinkan." << endl;
                    continue; // Lewati iterasi pada pengulangan
                }
                hasil = angka1 / angka2;
                break;
            default :
                cout << "Operasi tidak valid. Silakan coba lagi." << endl;
                continue; // Lewati iterasi pada pengulangan
        }

        // Tampilkan hasil
        cout << "Hasil : " << angka1 << " " << operasi << " " << angka2 << " = " << hasil << endl;
    }

    return 0;
}