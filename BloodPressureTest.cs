using System;
using System.Runtime.InteropServices;

public class BloodPressureDllTest
{
    [DllImport("CppWrapper.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int InitializeBP([MarshalAs(UnmanagedType.LPStr)] string modelDir);

    public static void Main(string[] args)
    {
        try
        {
            // C++ラッパーDLL経由で初期化
            int result = InitializeBP("models");
            Console.WriteLine($"InitializeBP result: {result}");
            // 既存のCython DLL直呼び出し部分はコメントアウト
            // TestBalancedDLL();
            Console.WriteLine("\nTest execution completed.");
            Environment.Exit(0);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\nApplication fatal error: {ex.Message}");
            Environment.Exit(1);
        }
    }
}
